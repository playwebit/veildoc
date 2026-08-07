"""Command-line interface.

Usage:
    veildoc ask paper.pdf "Can you help me improve the clarity of the methodology?"
    veildoc ask paper.pdf "..." --llm anthropic --api-key sk-...
    veildoc ask paper.pdf "..." --show-only    # never sends anywhere, just prints the redacted excerpt
"""

from __future__ import annotations

import argparse
import sys


def _build_embedding_backend():
    from .embeddings import LocalBGEEmbedding
    return LocalBGEEmbedding()


def _build_llm_backend(name: str | None, api_key: str | None, model: str | None):
    if name is None:
        return None
    if api_key is None and name != "ollama":
        raise SystemExit(f"--api-key is required for --llm {name}")

    if name == "anthropic":
        from .llm import AnthropicBackend
        return AnthropicBackend(api_key=api_key, model=model or "claude-sonnet-5")
    if name == "openai":
        from .llm import OpenAIBackend
        if model is None:
            raise SystemExit("--model is required for --llm openai")
        return OpenAIBackend(api_key=api_key, model=model)
    if name == "gemini":
        from .llm import GeminiBackend
        return GeminiBackend(api_key=api_key, model=model or "gemini-3.5-flash")
    if name == "ollama":
        from .llm import OllamaBackend
        return OllamaBackend(model=model or "llama3")
    raise SystemExit(f"Unknown --llm backend: {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="veildoc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question about a document")
    ask_parser.add_argument("document", help="Path to the source document (PDF)")
    ask_parser.add_argument("query", help="The question to ask")
    ask_parser.add_argument("--llm", choices=["anthropic", "openai", "gemini", "ollama"], default=None)
    ask_parser.add_argument("--api-key", default=None)
    ask_parser.add_argument("--model", default=None)
    ask_parser.add_argument("--k", type=int, default=1, help="Number of chunks to retrieve")
    ask_parser.add_argument(
        "--show-only", action="store_true",
        help="Only print the redacted excerpt; never send it anywhere",
    )

    args = parser.parse_args(argv)

    if args.command == "ask":
        from .pipeline import Pipeline

        embedding_backend = _build_embedding_backend()
        llm_backend = None if args.show_only else _build_llm_backend(args.llm, args.api_key, args.model)

        pipeline = Pipeline(embedding_backend=embedding_backend, llm_backend=llm_backend, k=args.k)
        pipeline.load_document(args.document)
        result = pipeline.ask(args.query, send_to_llm=(llm_backend is not None))

        print(f"Exposure: {result.exposure_pct:.2f}% of document")
        print(f"Entities redacted: {result.entities_redacted}")
        print()
        print("--- Redacted excerpt (this is what would be transmitted) ---")
        print(result.redacted_excerpt)
        if result.restored_response:
            print()
            print("--- AI response (restored) ---")
            print(result.restored_response)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
