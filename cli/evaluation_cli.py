import argparse
import json

from lib.search_utils import (
    GOLDEN_DATASET_PATH
)
from lib.hybrid_search import (
    rrf_search_command
)

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_dataset = json.load(f)

    for item in golden_dataset["test_cases"]:
        results = rrf_search_command(item["query"], k=60, limit=limit)
        relevant_docs = item["relevant_docs"]
        # Evaluate precision@k and recall@k
        precision_at_k = sum(1 for r in results["results"] if r["title"] in relevant_docs) / limit
        recall_at_k = sum(1 for r in results["results"] if r["title"] in relevant_docs) / len(relevant_docs)
        f1_score = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    

        print(
            f"Query: {item['query']}\n"
            f"Precision@{limit}: {precision_at_k:.4f}\n"
            f"Recall@{limit}: {recall_at_k:.4f}\n"
            f"F1 Score: {f1_score:.4f}\n"
            f"Retrieved Titles: {', '.join(r['title'] for r in results['results'])}\n"
            f"Relevant Titles: {relevant_docs}\n"
        )




if __name__ == "__main__":
    main()