# FAISS Gemini Embedding System

A high-performance embedding storage and similarity search system using Google's Gemini embedding model and FAISS vector database, with Model Context Protocol (MCP) server support.

## Features

- 🚀 **Fast Embedding Generation**: Uses Google's Gemini `embedding-001` model
- 🔍 **Efficient Similarity Search**: FAISS IndexFlatIP for cosine similarity
- 💾 **Persistent Storage**: Automatic disk persistence with configurable paths
- 🔄 **Retry Logic**: Robust error handling with exponential backoff
- 🌐 **MCP Server**: Model Context Protocol server for easy integration
- 🧪 **Comprehensive Testing**: Full test suite with 95%+ coverage
- ⚡ **Async Support**: Full asynchronous API for high performance

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/faiss-gemini-embedding.git
cd faiss-gemini-embedding

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set up API Key

Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and set it as an environment variable:

```bash
# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Windows Command Prompt
set GEMINI_API_KEY=your-api-key-here
```

### 3. Run the Example

```bash
python example.py
```

### 4. Start the MCP Server

```bash
# stdio transport (for MCP-compatible clients)
python -m src.faiss_gemini.cli server

# HTTP transport (for Postman and other HTTP clients)
python http_mcp_server.py
```

## Usage

### Programmatic API

```python
import asyncio
from src.faiss_gemini import EmbeddingApp

async def main():
    # Initialize the app
    app = EmbeddingApp(
        api_key="your-gemini-api-key",
        persist_path="./my_embeddings",
        embedding_dimension=768
    )
    
    # Add some texts
    await app.add_text("The quick brown fox jumps over the lazy dog.")
    await app.add_text("Python is a great programming language.")
    
    # Search for similar texts
    results = await app.search_similar("programming languages", k=5)
    for result in results:
        print(f"Score: {result.score:.4f} - {result.text}")
    
    # Get statistics
    stats = app.get_stats()
    print(f"Total embeddings: {stats['total_embeddings']}")

asyncio.run(main())
```

### MCP Server

The system includes a Model Context Protocol server that exposes three main operations:

1. **embed_text**: Generate and store embeddings for text
2. **search_similar**: Find similar texts using cosine similarity  
3. **get_stats**: Get system statistics

#### Transport Options

The MCP server supports two transport types:

- **stdio** (default): For integration with MCP-compatible clients like Claude Desktop
- **HTTP**: For HTTP-based clients like Postman

Start the MCP server:

```bash
# stdio transport (default)
python -m src.faiss_gemini.cli server

# HTTP transport (for Postman)
python http_mcp_server.py
```

#### Connecting with Postman

When running the HTTP server, you can connect using Postman:

1. Start the server: `python http_mcp_server.py`
2. Open Postman and create requests to: `http://localhost:8000`
3. Available endpoints:
   - `GET /` - Server info
   - `GET /tools` - List available tools
   - `POST /embed` - Embed text (send `{"text": "your text here"}`)
   - `POST /search` - Search similar texts (send `{"query": "search term", "k": 5}`)
   - `GET /stats` - Get server statistics
   - `POST /mcp/*` - MCP protocol endpoints

**Example Postman requests:**

**Embed text:**
```
POST http://localhost:8000/embed
Content-Type: application/json

{
  "text": "Hello world"
}
```

**Search similar texts:**
```
POST http://localhost:8000/search
Content-Type: application/json

{
  "query": "hello",
  "k": 3
}
```

### Command Line Interface

```bash
# Run MCP server with stdio transport (default)
python -m src.faiss_gemini.cli server

# Run HTTP server (for Postman)
python http_mcp_server.py

# Test embedding functionality
python -m src.faiss_gemini.cli test --text "Hello world"

# Show help
python -m src.faiss_gemini.cli --help
```

## Configuration

Configure the system using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | **Required** |
| `PERSIST_PATH` | Path to store embeddings | `./data/vector_store` |
| `EMBEDDING_DIMENSION` | Embedding dimension | `768` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Server    │    │  EmbeddingApp   │    │ Vector Store    │
│                 │───▶│                 │───▶│     (FAISS)     │
│ - embed_text    │    │ - add_text      │    │ - IndexFlatIP   │
│ - search_similar│    │ - search_similar│    │ - Persistence   │
│ - get_stats     │    │ - get_stats     │    │ - Cosine Sim    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Embedding Service│
                       │                 │
                       │ - Gemini API    │
                       │ - Retry Logic   │
                       │ - Batch Support │
                       └─────────────────┘
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_vector_store.py -v
```

## Development

### Code Formatting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Security Checks

```bash
# Check for security issues
bandit -r src/
safety check
```

## Performance

- **Embedding Generation**: ~100-500ms per text (depends on text length and API latency)
- **Similarity Search**: Sub-millisecond for thousands of embeddings
- **Storage**: Efficient binary format with automatic compression
- **Memory**: Scales linearly with number of embeddings

## Limitations

- Requires internet connection for embedding generation (Gemini API)
- Memory usage scales with number of stored embeddings
- API rate limits apply (check Google's documentation)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Format code (`black . && isort .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [Google Gemini](https://ai.google.dev/) - Embedding model
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification
