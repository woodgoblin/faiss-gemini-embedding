"""
MCP Server for FAISS Gemini Embedding System

Provides Model Context Protocol server interface for embedding and search operations.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    Implementation,
    InitializeRequestParams,
    InitializeResult,
    ListToolsResult,
    ServerCapabilities,
    TextContent,
    Tool,
    ToolsCapability,
)
from pydantic import BaseModel

from .app import EmbeddingApp
from .config import Config

logger = logging.getLogger(__name__)


class EmbedTextParams(BaseModel):
    """Parameters for embed_text tool."""

    text: str


class SearchSimilarParams(BaseModel):
    """Parameters for search_similar tool."""

    query: str
    k: Optional[int] = 5


class GetStatsParams(BaseModel):
    """Parameters for get_stats tool (no parameters needed)."""

    pass


class EmbeddingMCPServer:
    """MCP Server for embedding operations."""

    def __init__(self, config: Config):
        """
        Initialize the MCP server.

        Args:
            config: Application configuration
        """
        self.config = config
        self.app = EmbeddingApp(
            api_key=config.gemini_api_key,
            persist_path=config.persist_path,
            embedding_dimension=config.embedding_dimension,
        )
        self.server = Server("faiss-gemini-embedding")
        self._setup_handlers()

        logger.info("Initialized EmbeddingMCPServer")

    def _setup_handlers(self):
        """Set up MCP server handlers."""

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available tools."""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="embed_text",
                        description="Generate embedding for text and store it in the vector database",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text to embed and store",
                                }
                            },
                            "required": ["text"],
                        },
                    ),
                    Tool(
                        name="search_similar",
                        description="Search for similar texts using cosine similarity",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Query text to search for similar content",
                                },
                                "k": {
                                    "type": "integer",
                                    "description": "Number of similar results to return (default: 5)",
                                    "default": 5,
                                    "minimum": 1,
                                },
                            },
                            "required": ["query"],
                        },
                    ),
                    Tool(
                        name="get_stats",
                        description="Get statistics about the embedding system",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    ),
                ]
            )

        @self.server.call_tool()
        async def call_tool(params: CallToolRequestParams) -> CallToolResult:
            """Handle tool calls."""
            try:
                if params.name == "embed_text":
                    return await self._handle_embed_text(params.arguments or {})
                elif params.name == "search_similar":
                    return await self._handle_search_similar(params.arguments or {})
                elif params.name == "get_stats":
                    return await self._handle_get_stats(params.arguments or {})
                else:
                    raise ValueError(f"Unknown tool: {params.name}")

            except Exception as e:
                logger.error(f"Error handling tool call {params.name}: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")],
                    isError=True,
                )

    async def _handle_embed_text(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle embed_text tool call."""
        try:
            params = EmbedTextParams(**arguments)
            success = await self.app.add_text(params.text)

            if success:
                stats = self.app.get_stats()
                result_text = (
                    f"Successfully embedded and stored text.\n"
                    f"Text: {params.text[:100]}{'...' if len(params.text) > 100 else ''}\n"
                    f"Total embeddings in store: {stats['total_embeddings']}"
                )
            else:
                result_text = "Failed to embed and store text."

            return CallToolResult(content=[TextContent(type="text", text=result_text)])

        except Exception as e:
            logger.error(f"Error in embed_text: {e}")
            raise

    async def _handle_search_similar(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle search_similar tool call."""
        try:
            params = SearchSimilarParams(**arguments)
            results = await self.app.search_similar(params.query, params.k)

            if results:
                result_lines = [
                    f"Found {len(results)} similar texts for query: {params.query[:50]}{'...' if len(params.query) > 50 else ''}",
                    "",
                ]

                for i, result in enumerate(results, 1):
                    result_lines.append(f"{i}. Score: {result.score:.4f}")
                    result_lines.append(f"   Text: {result.text}")
                    result_lines.append("")

                result_text = "\n".join(result_lines)
            else:
                result_text = f"No similar texts found for query: {params.query}"

            return CallToolResult(content=[TextContent(type="text", text=result_text)])

        except Exception as e:
            logger.error(f"Error in search_similar: {e}")
            raise

    async def _handle_get_stats(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle get_stats tool call."""
        try:
            stats = self.app.get_stats()

            result_lines = [
                "Embedding System Statistics:",
                f"- Total embeddings: {stats['total_embeddings']}",
                f"- Embedding dimension: {stats['dimension']}",
                f"- Embedding model: {stats['embedding_model']}",
                f"- Index type: {stats['index_type']}",
                f"- Persist path: {stats['persist_path']}",
                f"- Status: {stats['status']}",
            ]

            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(result_lines))]
            )

        except Exception as e:
            logger.error(f"Error in get_stats: {e}")
            raise

    async def run(self, transport_type: str = "stdio"):
        """
        Run the MCP server.

        Args:
            transport_type: Transport type ("stdio", "ws", etc.)
        """
        logger.info(f"Starting MCP server with {transport_type} transport")

        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server

            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializeResult(
                        protocolVersion="2024-11-05",
                        capabilities=ServerCapabilities(
                            tools=ToolsCapability(listChanged=False)
                        ),
                        serverInfo=Implementation(
                            name="faiss-gemini-embedding", version="0.1.0"
                        ),
                    ),
                )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")


async def run_server(config: Optional[Config] = None):
    """
    Run the MCP server with the given configuration.

    Args:
        config: Configuration object. If None, loads from environment.
    """
    if config is None:
        from .config import get_config

        config = get_config()

    server = EmbeddingMCPServer(config)
    await server.run()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the server
    asyncio.run(run_server())
