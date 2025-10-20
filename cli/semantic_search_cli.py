#!/usr/bin/env python3

import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the semantic search model loads correctly")

    embed_parser =subparsers.add_parser("embed_text", help="Generate and display the embedding for a given text")
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify that the semantic search embeddings load correctly")

    embed_query = subparsers.add_parser("embedquery", help="Generate and display the embedding for a given query")
    embed_query.add_argument("query", type=str, help="Query text to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Search for similar documents given a query")
    search_parser.add_argument("query", type=str, help="Query text to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return (default: 5)")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to be chunked")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size of each chunk (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks (default: 0)")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text by sentence boundaries to preserve meaning")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to be semantically chunked")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of sentences per chunk (default: 4)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping sentences between chunks (default: 0)")

    subparsers.add_parser("embed_chunks", help="Generate and display the embedding for a given text chunk")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for similar document chunks given a query")
    search_chunked_parser.add_argument("query", type=str, help="Query text to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "verify":
            from lib.semantic_search import verify_model
            verify_model()
        
        case "embed_text":
            from lib.semantic_search import embed_text
            embed_text(args.text)

        case "verify_embeddings":
            from lib.semantic_search import verify_embeddings
            verify_embeddings()

        case "embedquery":
            from lib.semantic_search import embed_query_text
            embed_query_text(args.query)

        case "search":
            from lib.semantic_search import SemanticSearch, load_documents
            semantic_search = SemanticSearch()
            documents = load_documents()
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print("\n")
                print(f"{result['description']}")
                print("-------------------------------------")
                print("\n")

        case "chunk":
            from lib.chunk_utils import chunk_text
            chunk_text(args.text, chunk_size=args.chunk_size, overlap=args.overlap)
            
        case "semantic_chunk":
            from lib.chunk_utils import semantic_chunk
            semantic_chunk(args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap)

        case "embed_chunks":
            from lib.semantic_search import load_documents, ChunkedSemanticSearch
            documents = load_documents()
            print(f"Loaded documents for chunked embedding {len(documents)}")
            chunked_search = ChunkedSemanticSearch()
            #embeddings = chunked_search.build_chunk_embeddings(documents)
            embeddings = chunked_search.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            from lib.semantic_search import ChunkedSemanticSearch, load_documents
            documents = load_documents()
            chunked_search = ChunkedSemanticSearch()
            chunked_search.load_or_create_chunk_embeddings(documents)
            results = chunked_search.search_chunks(args.query, limit=args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                desc = result['description'].replace('\n', ' ').replace('\r', ' ')
                print(f"   {desc}...")
                print("-------------------------------------")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()