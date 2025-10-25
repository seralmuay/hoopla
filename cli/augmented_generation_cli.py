import argparse

from lib.hybrid_search import (
    rrf_search_command
)
from lib.augmented_generation import (
    augmented_generation
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rrf_search_command(
                args.query
            )
            # do RAG stuff here
            answer = augmented_generation(result["query"], result["results"])
            
            print("Search Results:")
            for doc in result["results"]:
                print(f"- {doc['title']}")
                print()
            print(f"RAG Response:")
            print(f"- {answer}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()