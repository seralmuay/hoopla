import argparse

from lib.describe_image import (
    describe_image
)


def main():
    parser = argparse.ArgumentParser(
        description="Describe an image based on a query"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query or description prompt for the image"
    )

    args = parser.parse_args()

    
    response = describe_image(args.image, args.query)
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
   
    

if __name__ == "__main__":
    main()