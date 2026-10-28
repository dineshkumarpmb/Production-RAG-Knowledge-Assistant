from types import SimpleNamespace as NS
from typing import Any

import anthropic
import httpx2
import pytest

from app.core.config import Settings
from app.generation.claude import ClaudeAnswerGenerator, LLMError
from app.generation.conversation import REWRITE_SYSTEM, SUMMARY_SYSTEM, ConversationHelper
from app.generation.prompts import HistoryMessage, build_messages
from app.generation.types import Completed, UsageUpdate
from tests.unit.test_generation import FakeClient, block, final, text_delta, text_start

HISTORY = [
    HistoryMessage("user", "What is the hotel limit?"),
    HistoryMessage("assistant", "Hotels are capped at 150 EUR per night."),
]


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response, self._error = response, error

    async def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if self._error:
            raise self._error
        return self._response


def reply(text: str, stop_reason: str = "end_turn") -> NS:
    return NS(
        content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
        stop_reason=stop_reason,
        model="claude-haiku-5-5",
        usage=NS(
            input_tokens=200,
            output_tokens=30,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=None,
        ),
    )


def helper(settings: Settings, messages: FakeMessages) -> ConversationHelper:
    return ConversationHelper(NS(messages=messages), settings)  # type: ignore[arg-type]


async def test_rewrite_request_and_result(settings: Settings) -> None:
    messages = FakeMessages(reply('  "Hotel limit for contractors"\n'))
    result = await helper(settings, messages).rewrite("what about contractors?", HISTORY)

    assert result.text == "Hotel limit for contractors"
    # 200 x $0.10 + 30 x $0.50 per million tokens (Haiku 5.5)
    assert result.usage.cost_usd == pytest.approx((200 * 0.10 + 30 * 0.50) / 1_000_000)
    (call,) = messages.calls
    assert call["model"] == "claude-haiku-5-5"
    assert call["system"] == REWRITE_SYSTEM
    assert call["output_config"] == {"effort": "low"}
    # Haiku 5.5 has no server-side refusal fallback, and sampling parameters are rejected.
    assert not {"fallbacks", "betas", "temperature", "thinking"} & set(call)
    prompt = call["messages"][0]["content"]
    assert "User: What is the hotel limit?" in prompt
    assert "Assistant: Hotels are capped" in prompt
    assert "<latest_message>\nwhat about contractors?\n</latest_message>" in prompt


async def test_an_answer_instead_of_a_query_is_not_used(settings: Settings) -> None:
    rambling = "Contractors " * 100
    result = await helper(settings, FakeMessages(reply(rambling))).rewrite(
        "and contractors?", HISTORY
    )
    assert result.text == "and contractors?"


@pytest.mark.parametrize(
    ("response", "error", "code"),
    [
        (reply("x", stop_reason="refusal"), None, "llm_refused"),
        (reply("   "), None, "llm_empty"),
        (
            None,
            anthropic.RateLimitError(
                "slow down",
                response=httpx2.Response(429, request=httpx2.Request("POST", "https://x")),
                body=None,
            ),
            "llm_rate_limited",
        ),
        (None, TypeError("Could not resolve authentication method."), "llm_not_configured"),
    ],
)
async def test_helper_failures_raise_llm_errors(
    settings: Settings, response: Any, error: Exception | None, code: str
) -> None:
    with pytest.raises(LLMError) as err:
        await helper(settings, FakeMessages(response, error)).rewrite("q", HISTORY)
    assert err.value.code == code


async def test_summarize_includes_the_previous_summary(settings: Settings) -> None:
    messages = FakeMessages(reply("The user asked about hotel limits (150 EUR)."))
    result = await helper(settings, messages).summarize("Earlier: travel booking.", HISTORY)
    assert result.text.startswith("The user asked")
    (call,) = messages.calls
    assert call["system"] == SUMMARY_SYSTEM
    assert (
        "<previous_summary>\nEarlier: travel booking.\n</previous_summary>"
        in call["messages"][0]["content"]
    )


def test_build_messages_with_history_and_summary() -> None:
    history = [HistoryMessage("assistant", "orphaned answer"), *HISTORY]
    messages = build_messages("And in Paris?", [block()], history, summary="Talked about travel.")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]  # must start with user
    assert messages[0]["content"] == "What is the hotel limit?"
    current = messages[-1]["content"]
    assert current[0] == {
        "type": "text",
        "text": "Summary of the earlier part of this conversation:\nTalked about travel.",
    }
    assert [c["type"] for c in current[1:]] == ["document", "text"]
    assert current[-1]["text"] == "And in Paris?"
    assert build_messages("Q", [block()])[0]["role"] == "user"  # no history: one turn


async def test_usage_updates_are_emitted_while_streaming(settings: Settings) -> None:
    usage = NS(
        input_tokens=1000,
        output_tokens=1,
        cache_read_input_tokens=600,
        cache_creation_input_tokens=0,
    )
    client = FakeClient(
        [
            NS(type="message_start", message=NS(model="claude-opus-5-5", usage=usage)),
            text_start(),
            text_delta("Hello"),
            NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=NS(output_tokens=50)),
        ],
        final(),
    )
    events = [e async for e in ClaudeAnswerGenerator(client, settings).stream("Q", [block()])]
    updates = [e for e in events if isinstance(e, UsageUpdate)]
    assert [u.usage.output_tokens for u in updates] == [1, 50]
    assert updates[-1].usage.cost_usd == pytest.approx((1000 * 4 + 50 * 20 + 600 * 0.2) / 1_000_000)
    assert isinstance(events[-1], Completed)
