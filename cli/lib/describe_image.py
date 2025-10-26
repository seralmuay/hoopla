import os
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def describe_image(image_path: str, query: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    if mime is None:
        raise ValueError(f"Unsupported file type: {image_path}")
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()

    
    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary"""
    parts = [
        prompt,
        types.Part.from_bytes(data=image_content, mime_type=mime),
        query.strip(),
    ]
    response = client.models.generate_content(model=model, contents=parts)
    return response

