"""Upload a folder of documents through the running API and wait until they are processed.

    python -m app.evaluation.load_corpus ../eval/corpus/sample --api http://localhost:8000

Uses the normal upload path (validation, storage, worker), so it needs `docker compose up`.
Re-running is safe: files already uploaded come back as duplicates.
"""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from app.ingestion.validation import EXTENSIONS

_DONE = {"ready", "failed"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for processing")
    args = parser.parse_args(argv)

    files = sorted(p for p in args.directory.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        print(f"No supported files in {args.directory}", file=sys.stderr)
        return 1

    with httpx.Client(base_url=args.api, timeout=120) as client:
        pending: dict[str, str] = {}
        for path in files:
            response = client.post("/documents", files={"file": (path.name, path.read_bytes())})
            if response.status_code not in (200, 202):
                print(f"FAILED  {path.name}: {response.json().get('message', response.text)}")
                continue
            body = response.json()
            pending[body["document"]["id"]] = path.name
            print(f"{'exists ' if body['duplicate'] else 'queued '} {path.name}")

        failed = 0
        deadline = time.monotonic() + args.timeout
        while pending and time.monotonic() < deadline:
            for doc_id, name in list(pending.items()):
                doc = client.get(f"/documents/{doc_id}").json()
                if doc["status"] in _DONE:
                    del pending[doc_id]
                    if doc["status"] == "failed":
                        failed += 1
                        print(f"FAILED  {name}: {doc['error']}")
                    else:
                        chunks = doc["metadata"].get("chunking", {}).get("chunk_count")
                        print(f"ready   {name} ({doc['page_count']} pages, {chunks} chunks)")
            if pending:
                time.sleep(2)

    for name in pending.values():
        print(f"TIMEOUT {name}")
    return 1 if failed or pending else 0


if __name__ == "__main__":
    sys.exit(main())
