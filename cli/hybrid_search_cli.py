import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:

    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", nargs='+', type=float, help="List of scores to normalize")

    weighted_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor (alpha)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="RRF k parameter (default: 60)")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 10)")
    rrf_search_parser.add_argument("--enhance", type=str, choices=['spell','rewrite','expand'], help="Enhancement method to apply to the query")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=['individual','batch'], help="Reranking method to use")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                print("No scores provided for normalization.")
                return
            
            from cli.lib.hybrid_search import HybridSearch
            normalized_scores = HybridSearch.normalize_scores(scores)
            
            for score in normalized_scores:
                print(f"* {score:.4f}")
        
        case "weighted-search":

            from cli.lib.hybrid_search import HybridSearch
            from cli.lib.semantic_search import load_documents
            documents = load_documents()
            query = args.query
            alpha = args.alpha
            limit = args.limit

            hybrid_search = HybridSearch(documents)
            result =hybrid_search.weighted_search(query, alpha, limit)
            for i,res in enumerate(result, start=1):
                #print(f"* {res['hybrid_score']:.4f} | {res['document']['title']} (ID: {res['id']}) | BM25: {res['bm25_score']:.4f} | Semantic: {res['semantic_score']:.4f}")
                print(f"{i}. {res['document']['title']}")
                print(f"Hybrid Score: {res['hybrid_score']:.4f}")
                print(f"BM25 Score: {res['bm25_score']:.4f} | Semantic Score: {res['semantic_score']:.4f}")
                print(res['document']['description'][:200] + "...")
                print("\n")

        case "rrf-search":
            from cli.lib.hybrid_search import HybridSearch
            from cli.lib.semantic_search import load_documents
            documents = load_documents()
            query = args.query
            enhanced_query = ""
            k = args.k
            limit = args.limit

            if args.enhance == 'spell':
                enhanced_query = HybridSearch.spell_enhance_query(query)
                print( f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
            elif args.enhance == 'rewrite':
                enhanced_query = HybridSearch.rewrite_enhance_query(query)
                print( f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
            elif args.enhance == 'expand':
                enhanced_query = HybridSearch.expand_enhance_query(query)
                print( f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")

            hybrid_search = HybridSearch(documents)
            if args.rerank_method in ['individual', 'batch']:
                print("Fetching extra candidates for LLM reranking...\n")
                result = hybrid_search.rrf_search(query if enhanced_query == "" else enhanced_query, k, limit*5, candidate_multiplier=5)
            else:
                result = hybrid_search.rrf_search(query if enhanced_query == "" else enhanced_query, k, limit)

            if args.rerank_method == 'individual':
                print("Applying individual LLM reranking...\n")
                results = HybridSearch.llm_ranking(query if enhanced_query == "" else enhanced_query, limit, result)
            elif args.rerank_method == 'batch':
                print("Applying batch LLM reranking...\n")
                results = HybridSearch.llm_ranking_batch(query if enhanced_query == "" else enhanced_query, limit, result)

            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['document']['title']}")
                #print(f"Rerank Score: {res['llm_score']:.3f} / 10.0")
                print(f"RRF Score: {res['rrf_score']:.3f}")
                print(f"BM25 Rank: {res['bm25_rank']} | Semantic Rank: {res['semantic_rank']}")
                print(res['document']['description'][:200] + "...")
                print("\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
