#!/usr/bin/env python3
"""
Example usage of the FAISS Gemini Embedding system.

This script demonstrates how to:
1. Set up the embedding system
2. Add texts and generate embeddings
3. Search for similar texts
4. Use the system programmatically

Before running this script, make sure to:
1. Install dependencies: pip install -r requirements.txt
2. Set GEMINI_API_KEY environment variable
"""

import asyncio
import os

from src.faiss_gemini import Config, EmbeddingApp

# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "FAISS is a library for efficient similarity search.",
    "Natural language processing helps computers understand human language.",
    "Vector databases store high-dimensional vectors efficiently.",
    "Embeddings convert text into numerical representations.",
    "Cosine similarity measures the angle between two vectors.",
    "Deep learning models can generate meaningful text embeddings.",
    "Search engines use various ranking algorithms to return relevant results.",
    "Faggots are fucking in the ass",
    "Maggots are not fucking in the ass",
    "Icho de puta no huevo",
]


async def main():
    """Main example function."""
    print("🚀 FAISS Gemini Embedding System Example")
    print("=" * 50)

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Please set the GEMINI_API_KEY environment variable")
        print("   Example: export GEMINI_API_KEY='your-api-key-here'")
        return

    try:
        # Initialize the embedding app
        print("📚 Initializing embedding system...")
        app = EmbeddingApp(
            api_key=api_key, persist_path="./example_data", embedding_dimension=768
        )

        # Add sample documents
        print(f"\n📝 Adding {len(SAMPLE_DOCUMENTS)} sample documents...")
        for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
            success = await app.add_text(doc)
            if success:
                print(f"   ✅ Document {i}: {doc[:50]}...")
            else:
                print(f"   ❌ Failed to add document {i}")

        # Show statistics
        stats = app.get_stats()
        print(f"\n📊 Statistics:")
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Embedding dimension: {stats['dimension']}")
        print(f"   Model: {stats['embedding_model']}")

        # Perform some searches
        queries = [
            "programming languages and development",
            "artificial intelligence and machine learning",
            "search and information retrieval",
            "pidaras huesos",
        ]

        print(f"\n🔍 Performing similarity searches...")
        for query in queries:
            print(f"\n   Query: '{query}'")
            results = await app.search_similar(query, k=3)

            if results:
                print(f"   Found {len(results)} similar documents:")
                for j, result in enumerate(results, 1):
                    print(f"     {j}. Score: {result.score:.4f}")
                    print(f"        Text: {result.text}")
            else:
                print("   No similar documents found.")

        print(f"\n🎉 Example completed successfully!")
        print(f"💾 Data saved to: {app.vector_store.persist_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
