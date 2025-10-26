import argparse

from lib.hybrid_search import (
    rrf_search_command
)
from lib.augmented_generation import (
    augmented_generation,
    augmented_summarization,
    augmented_citations,
    augmented_question
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize the RAG"
    )
    summarize_parser.add_argument("query", type=str, help="Query to the RAG")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    citations_parser = subparsers.add_parser(
        "citations", help="Get citations for the RAG"
    )
    citations_parser.add_argument("query", type=str, help="Query to the RAG")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question based on the RAG"
    )
    question_parser.add_argument("query", type=str, help="Query to the RAG")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

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

        case "summarize":
            result = rrf_search_command(
                args.query
            )
            
            answer = augmented_summarization(result["query"], result["results"])
            
            print("Search Results:")
            for doc in result["results"]:
                print(f"- {doc['title']}")
                print()
            print(f"LLM Summary:")
            print(f"- {answer}")

        case "citations":
            result = rrf_search_command(
                args.query
            )
            citations = augmented_citations(result["query"], result["results"])
            
            print("Search Results:")
            for doc in result["results"]:
                print(f"- {doc['title']}")
                print()
            print(f"LLM Answer:")
            print(f"- {citations}")

        case "question":
            result = rrf_search_command(
                args.query
            )
            question = augmented_question(result["query"], result["results"])
            print("Search Results:")
            for doc in result["results"]:
                print(f"- {doc['title']}")
                print()
            print(f"LLM Answer:")
            print(f"- {question}")
            

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()