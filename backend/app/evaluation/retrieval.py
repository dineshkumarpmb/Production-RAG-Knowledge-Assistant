"""Retrieval-only evaluation: Recall@k, MRR, nDCG@10 and latency per search configuration
(keyword, vector, hybrid, hybrid+rerank).

Runs against the configured database and embedding provider (the same ones the API uses):

    python -m app.evaluation.retrieval --dataset ../eval/datasets/sample_qa.jsonl

Dataset: one JSON object per line,
    {"id": "q1", "question": "...", "type": "exact-term",
     "gold": [{"filename": "security.md", "page": 1, "heading": "Incident codes"}],
     "user_email": "optional; defaults to --user-email"}
Questions with an empty "gold" list are unanswerable. They are left out of the ranking metrics;
with reranking, their best scores feed a table showing how a RERANK_MIN_SCORE threshold would
trade correct refusals against wrongly refused answerable questions.
"""

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.db.access import Principal
from app.db.models import User
from app.db.session import get_async_sessionmaker
from app.embeddings import get_embedding_provider
from app.evaluation.metrics import (
    GoldRef,
    RetrievedChunk,
    gains,
    hit_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from app.rerank import get_reranker
from app.retrieval.service import HybridRetriever, RetrievalConfig
from app.retrieval.types import Mode

DEFAULT_KS = (1, 5, 10, 20)
THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class EvalConfig:
    label: str
    mode: Mode
    rerank: bool = False


CONFIGS = {
    c.label: c
    for c in (
        EvalConfig("keyword", Mode.KEYWORD),
        EvalConfig("vector", Mode.VECTOR),
        EvalConfig("hybrid", Mode.HYBRID),
        EvalConfig("hybrid+rerank", Mode.HYBRID, rerank=True),
    )
}


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold: tuple[GoldRef, ...]
    type: str = "unspecified"
    user_email: str | None = None


@dataclass
class QuestionResult:
    id: str
    type: str
    config: str
    first_relevant_rank: int | None
    metrics: dict[str, float]
    latency_ms: float
    degraded: list[str]
    rerank_ms: float | None = None
    best_rerank_score: float | None = None


@dataclass
class ConfigSummary:
    config: str
    questions: int
    metrics: dict[str, float]
    by_type: dict[str, dict[str, float]]
    latency_p50_ms: float
    latency_p95_ms: float
    degraded_queries: int
    rerank_p50_ms: float | None = None
    rerank_p95_ms: float | None = None


@dataclass
class Report:
    dataset: str
    created_at: str
    git_sha: str | None
    config: dict[str, Any]
    answerable: int
    unanswerable: int
    summaries: list[ConfigSummary]
    results: list[QuestionResult] = field(default_factory=list)
    # Per candidate RERANK_MIN_SCORE: share of unanswerable questions correctly refused and
    # of answerable questions wrongly refused (best reranked score below the threshold).
    thresholds: list[dict[str, float]] = field(default_factory=list)


def load_dataset(path: Path) -> list[Question]:
    questions = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            questions.append(
                Question(
                    id=str(raw["id"]),
                    question=raw["question"],
                    gold=tuple(GoldRef(**g) for g in raw.get("gold", [])),
                    type=raw.get("type", "unspecified"),
                    user_email=raw.get("user_email"),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid question: {exc}") from exc
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate question ids")
    return questions


def score(
    ranking: Sequence[RetrievedChunk], gold: Sequence[GoldRef], ks: Sequence[int]
) -> dict[str, float]:
    metrics = {f"recall@{k}": recall_at_k(ranking, gold, k) for k in ks}
    metrics |= {f"hit@{k}": hit_at_k(ranking, gold, k) for k in ks}
    metrics["mrr"] = reciprocal_rank(ranking, gold)
    metrics["ndcg@10"] = ndcg_at_k(ranking, gold, 10)
    return metrics


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return (
        {key: round(statistics.fmean(r[key] for r in rows), 4) for key in rows[0]} if rows else {}
    )


async def _principals(retriever_sessions: Any, emails: set[str]) -> dict[str, Principal]:
    async with retriever_sessions() as session:
        users = (await session.scalars(select(User).where(User.email.in_(emails)))).all()
    found = {u.email: Principal.of(u) for u in users}
    if missing := emails - set(found):
        raise ValueError(f"Unknown users in dataset: {', '.join(sorted(missing))}")
    return found


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


async def evaluate(
    questions: Sequence[Question],
    retriever: HybridRetriever,
    principals: dict[str, Principal],
    *,
    default_email: str,
    configs: Sequence[EvalConfig],
    ks: Sequence[int] = DEFAULT_KS,
    dataset_name: str = "",
) -> Report:
    answerable = [q for q in questions if q.gold]
    results: list[QuestionResult] = []
    summaries: list[ConfigSummary] = []
    thresholds: list[dict[str, float]] = []
    for config in configs:
        rows: list[QuestionResult] = []
        refusal_scores: list[float] = []  # best reranked score per unanswerable question
        for q in questions:
            if not q.gold and not config.rerank:
                continue
            result = await retriever.search(
                principals[q.user_email or default_email],
                q.question,
                mode=config.mode,
                top_k=max(ks),
                collapse_duplicates=False,  # a duplicate may be the gold chunk's twin
                rerank=config.rerank,
            )
            best = result.hits[0].rerank_score if result.reranker_model and result.hits else None
            if not q.gold:
                refusal_scores.append(best or 0.0)
                continue
            ranking = [
                RetrievedChunk(h.filename, h.page_start, h.page_end, tuple(h.heading_path))
                for h in result.hits
            ]
            first = next((i for i, g in enumerate(gains(ranking, q.gold), 1) if g), None)
            rows.append(
                QuestionResult(
                    id=q.id,
                    type=q.type,
                    config=config.label,
                    first_relevant_rank=first,
                    metrics=score(ranking, q.gold, ks),
                    latency_ms=result.timings_ms.get("total", 0.0),
                    degraded=result.degraded,
                    rerank_ms=result.timings_ms.get("rerank"),
                    best_rerank_score=best,
                )
            )
        summaries.append(_summarise(config, rows))
        results.extend(rows)
        answer_scores = [r.best_rerank_score for r in rows if r.best_rerank_score is not None]
        if config.rerank and not thresholds and answer_scores and refusal_scores:
            thresholds = [
                {
                    "threshold": t,
                    "unanswerable_refused": round(
                        sum(s < t for s in refusal_scores) / len(refusal_scores), 3
                    ),
                    "answerable_refused": round(
                        sum(s < t for s in answer_scores) / len(answer_scores), 3
                    ),
                }
                for t in THRESHOLDS
            ]

    return Report(
        dataset=dataset_name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        config=asdict(retriever.config) | {"ks": list(ks)},
        answerable=len(answerable),
        unanswerable=len(questions) - len(answerable),
        summaries=summaries,
        results=results,
        thresholds=thresholds,
    )


def _summarise(config: EvalConfig, rows: list[QuestionResult]) -> ConfigSummary:
    latencies = [r.latency_ms for r in rows]
    rerank = [r.rerank_ms for r in rows if r.rerank_ms is not None]
    types = sorted({r.type for r in rows})
    return ConfigSummary(
        config=config.label,
        questions=len(rows),
        metrics=_mean([r.metrics for r in rows]),
        by_type={t: _mean([r.metrics for r in rows if r.type == t]) for t in types},
        latency_p50_ms=round(percentile(latencies, 50), 1),
        latency_p95_ms=round(percentile(latencies, 95), 1),
        degraded_queries=sum(1 for r in rows if r.degraded),
        rerank_p50_ms=round(percentile(rerank, 50), 1) if rerank else None,
        rerank_p95_ms=round(percentile(rerank, 95), 1) if rerank else None,
    )


def render_markdown(report: Report, ks: Sequence[int]) -> str:
    cols = [f"recall@{k}" for k in ks] + ["mrr", "ndcg@10"]
    lines = [
        f"# Retrieval evaluation: {report.dataset}",
        "",
        f"- Run: {report.created_at} (git {report.git_sha or 'unknown'})",
        f"- Questions: {report.answerable} answerable, {report.unanswerable} unanswerable",
        f"- Config: {json.dumps(report.config)}",
        "",
        "| Config | " + " | ".join(cols) + " | p50 ms | p95 ms | Rerank p50 ms | Degraded |",
        "|------|" + "|".join("---:" for _ in cols) + "|---:|---:|---:|---:|",
    ]
    for s in report.summaries:
        values = " | ".join(f"{s.metrics.get(c, 0):.3f}" for c in cols)
        latency = f"{s.latency_p50_ms} | {s.latency_p95_ms} | {s.rerank_p50_ms or '-'}"
        lines.append(f"| {s.config} | {values} | {latency} | {s.degraded_queries} |")
    lines += ["", "## Recall@5 by question type", ""]
    types = sorted({t for s in report.summaries for t in s.by_type})
    lines.append("| Type | " + " | ".join(s.config for s in report.summaries) + " |")
    lines.append("|------|" + "|".join("---:" for _ in report.summaries) + "|")
    for t in types:
        row = " | ".join(f"{s.by_type.get(t, {}).get('recall@5', 0):.3f}" for s in report.summaries)
        lines.append(f"| {t} | {row} |")
    if report.thresholds:
        lines += [
            "",
            "## Rerank score threshold (RERANK_MIN_SCORE)",
            "",
            "| Threshold | Unanswerable correctly refused | Answerable wrongly refused |",
            "|---:|---:|---:|",
        ]
        lines += [
            f"| {t['threshold']} | {t['unanswerable_refused']:.0%} "
            f"| {t['answerable_refused']:.0%} |"
            for t in report.thresholds
        ]
    misses = [r for r in report.results if r.first_relevant_rank is None]
    if misses:
        lines += ["", "## Questions with no relevant chunk in the top results", ""]
        lines += [f"- `{r.id}` ({r.type}, {r.config})" for r in misses]
    return "\n".join(lines) + "\n"


async def _main(args: argparse.Namespace, settings: Settings) -> Report:
    questions = load_dataset(args.dataset)
    sessions = get_async_sessionmaker()
    retriever = HybridRetriever(
        sessions,
        partial(get_embedding_provider, settings),
        RetrievalConfig.from_settings(settings),
        reranker_factory=partial(get_reranker, settings),
    )
    emails = {q.user_email or args.user_email for q in questions}
    principals = await _principals(sessions, emails)
    return await evaluate(
        questions,
        retriever,
        principals,
        default_email=args.user_email,
        configs=[CONFIGS[c] for c in args.configs],
        ks=args.k,
        dataset_name=args.dataset.name,
    )


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument(
        "--user-email",
        required=True,
        help="search as this user (who must be able to read the corpus)",
    )
    parser.add_argument("--out", type=Path, default=Path("../eval/reports"))
    args = parser.parse_args(argv)

    if sys.platform == "win32":  # psycopg's async mode needs a selector event loop on Windows
        report = asyncio.run(_main(args, settings), loop_factory=asyncio.SelectorEventLoop)
    else:
        report = asyncio.run(_main(args, settings))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = args.out / f"{stamp}-{report.git_sha or 'nogit'}-retrieval"
    markdown = render_markdown(report, args.k)
    base.with_suffix(".md").write_text(markdown, encoding="utf-8")
    base.with_suffix(".json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(markdown)
    print(f"Wrote {base}.md and .json")


if __name__ == "__main__":
    main()
