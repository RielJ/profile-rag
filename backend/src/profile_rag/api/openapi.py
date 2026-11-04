"""Write the API's OpenAPI schema as JSON, without starting a server.

    uv run python -m profile_rag.api.openapi PATH

`make types` runs this and turns the file into frontend/src/api/schema.d.ts
with openapi-typescript. The schema comes from the route and model
definitions only: building it opens no database, Ollama, or Pi.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any


def openapi_schema() -> dict[str, Any]:
    # The app turns tracing on at creation when LANGSMITH_TRACING=true. The
    # schema needs no tracing, and a refused tracing setup must not block it.
    os.environ["LANGSMITH_TRACING"] = "false"
    from profile_rag.api.app import create_app

    return create_app().openapi()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m profile_rag.api.openapi PATH", file=sys.stderr)
        return 2
    text = json.dumps(openapi_schema(), indent=2, ensure_ascii=False) + "\n"
    Path(argv[0]).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
