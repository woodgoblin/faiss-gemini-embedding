"""
Command-line interface for the FAISS Gemini Embedding system.
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

from .app import EmbeddingApp
from .config import Config, get_config
from .mcp_server import run_server


def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def run_mcp_server(args):
    """Run the MCP server."""
    print("Starting FAISS Gemini Embedding MCP Server...")
    print("Make sure you have set the GEMINI_API_KEY environment variable.")
    print("Press Ctrl+C to stop the server.")

    try:
        config = get_config()
        await run_server(config)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error running server: {e}")
        sys.exit(1)


async def test_embedding(args):
    """Test embedding functionality."""
    try:
        config = get_config()
        app = EmbeddingApp(
            api_key=config.gemini_api_key,
            persist_path=config.persist_path,
            embedding_dimension=config.embedding_dimension,
        )

        test_text = args.text or "This is a test document for embedding."
        print(f"Testing embedding for: {test_text}")

        # Add the text
        success = await app.add_text(test_text)
        if success:
            print("✅ Successfully embedded and stored text!")

            # Search for similar texts
            print(f"\nSearching for similar texts...")
            results = await app.search_similar(test_text, k=3)

            if results:
                print(f"Found {len(results)} similar texts:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Score: {result.score:.4f} - {result.text[:100]}...")
            else:
                print("No similar texts found.")

            # Show stats
            stats = app.get_stats()
            print(f"\nStats: {stats['total_embeddings']} embeddings in store")

        else:
            print("❌ Failed to embed text. Check your API key and network connection.")

    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="FAISS Gemini Embedding System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MCP server (most common usage)
  python -m src.faiss_gemini.cli server
  
  # Test embedding functionality
  python -m src.faiss_gemini.cli test --text "Hello world"
  
Environment variables:
  GEMINI_API_KEY        Google Gemini API key (required)
  PERSIST_PATH          Path to store embeddings (default: ./data/vector_store)
  EMBEDDING_DIMENSION   Embedding dimension (default: 768)
  LOG_LEVEL             Logging level (default: INFO)
        """,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run the MCP server")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test embedding functionality")
    test_parser.add_argument("--text", type=str, help="Text to test embedding with")

    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    if args.command == "server":
        await run_mcp_server(args)
    elif args.command == "test":
        await test_embedding(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
