#!/usr/bin/env python3

import argparse
from inverted_index import InvertedIndex
from text_utils import TextUtils
import math

def main() -> None:

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF")

    args = parser.parse_args()
    text_utils = TextUtils()
    
    match args.command:
        case "search":
            results = []
            # print the search query here
            print(f"Searching for: {args.query}")
            query_tokens = text_utils.tokenize(args.query)

            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as e:
                print(e)
                return
            
            for query_token in query_tokens:
                docs = index.get_documents(query_token)
                for doc in docs:
                    if doc not in results:
                        results.append(doc)
                    if len(results) == 5:
                        break
                if len(results) == 5:
                    break

            for movie in results:
                print(f"ID: {movie['id']}, Title: {movie['title']}")

        case "build":
            index = InvertedIndex()
            index.build()
            index.save()

        case "tf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(f"Term frequency for doc_id={args.doc_id}, term='{args.term}': {tf}")

        case "idf":
            index = InvertedIndex()
            index.load()
            # documents = index.get_documents(args.term)
            document_count = len(index.docmap)
            documents = len(index.index.get(args.term, set()))

            print(f"documents:{documents}")
            # Using smoothed IDF formula
            # idf = math.log((document_count + 1) / (len(documents) + 1))
            idf = round(math.log((document_count + 1) / (documents + 1)), 2)

            print(f"Inverse document frequency for term '{args.term}': {idf}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()