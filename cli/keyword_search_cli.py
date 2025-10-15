#!/usr/bin/env python3

import argparse
from inverted_index import InvertedIndex
from text_utils import TextUtils
from search_utils import BM25_K1, BM25_B

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

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 B parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

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
            idf = index.get_idf(args.term)

            print(f"Inverse document frequency for term '{args.term}': {idf:.2f}")

        case "tfidf":
            index = InvertedIndex()
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            idf = index.get_idf(args.term)
            tfidf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")

        case "bm25idf":
            index = InvertedIndex()
            index.load()
            try:
                bm25_idf = index.get_bm25_idf(args.term)
                print(f"BM25 IDF score for term '{args.term}': {bm25_idf:.2f}")
            except ValueError as e:
                print(e)

        case "bm25tf":
            index = InvertedIndex()
            index.load()
            bm25_tf = index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")

        case "bm25search":
            index = InvertedIndex()
            index.load()
            results = index.bm25_search(args.query, limit=5)
            for doc_id, score in results:
                movie = index.docmap.get(doc_id, {})
                print(f"({doc_id}) {movie.get('title', 'N/A')} - Score: {score:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()