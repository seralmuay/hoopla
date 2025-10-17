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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()