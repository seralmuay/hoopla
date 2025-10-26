import argparse

from lib.multimodal_search import (
    verify_image_embeddings,
    image_search_command
)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embeddings_parser =subparsers.add_parser(
        "verify_image_embedding", help="Verify image embeddings")
    verify_image_embeddings_parser.add_argument("image_path", type=str, help="Path to the image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search for movies using an image")
    image_search_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embeddings(args.image_path)
        case "image_search":
            results = image_search_command(args.image_path)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (similarity: {res['score']:.3f})")
                print(f"   {res['description'][:100]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()  