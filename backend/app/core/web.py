"""Security headers for every response."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_COMMON = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-site"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]
# The API returns JSON and files only, so nothing may load or execute.
_API = [
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
    (b"cache-control", b"no-store"),  # tokens and private documents must not be cached
]
_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")  # Swagger UI needs its CDN scripts


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        extra = _COMMON if scope["path"].startswith(_DOCS_PATHS) else _COMMON + _API

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {name.lower() for name, _ in message.get("headers", [])}
                message["headers"] = [
                    *message.get("headers", []),
                    *((n, v) for n, v in extra if n not in present),
                ]
            await send(message)

        await self.app(scope, receive, send_with_headers)
