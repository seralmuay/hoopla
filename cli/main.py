import os
import re
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from cli.lib.chunk_utils import chunk_text
# from cli.lib.hybrid_search import HybridSearch


# def add_vectors(vector1, vector2):
#     if len(vector1) != len(vector2):
#         raise ValueError("Vectors must be of the same length")
    
#     return [a + b for a, b in zip(vector1, vector2)]

# def substract_vectors(vector1, vector2):
#     if len(vector1) != len(vector2):
#         raise ValueError("Vectors must be of the same length")
    
#     return [a - b for a, b in zip(vector1, vector2)]

# def dot_product(vec1,vec2):
#     if len(vec1) != len(vec2):
#         raise ValueError("Vectors must be of the same length")
#     return sum(a*b for a,b in zip(vec1,vec2))
    
# def euclidean_norm(vec):
#     total = 0.0
#     for x in vec:
#         total += x**2

#     return total**0.5

# def cosine_similarity(vec1,vec2):
#     dot_prod = dot_product(vec1,vec2)
#     norm1 = euclidean_norm(vec1)
#     norm2 = euclidean_norm(vec2)

#     if norm1 == 0 or norm2 == 0:
#         return 0.0

#     return dot_prod / (norm1 * norm2)

# def semantic_chunk(
#     text: str,
#     max_chunk_size: int = 4,
#     overlap: int = 0,) -> list[str]:
#     sentences = re.split(r"(?<=[.!?])\s+", text)
#     chunks = []
#     i = 0
#     n_sentences = len(sentences)
#     while i < n_sentences - overlap:
#         chunk_sentences = sentences[i : i + max_chunk_size]
#         chunks.append(" ".join(chunk_sentences))
#         i += max_chunk_size - overlap
#     return chunks


# def semantic_chunk_text(
#     text: str,
#     max_chunk_size: int = 4,
#     overlap: int = 0,) -> None:
#     chunks = semantic_chunk(text, max_chunk_size, overlap)
#     print(f"Semantically chunking {len(text)} characters")
#     for i, chunk in enumerate(chunks):
#         print(f"{i + 1}. {chunk}")

def main() -> None: 
    from sentence_transformers.cross_encoder import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    scores = model.predict([["My first", "sentence pair"], ["Second text", "pair"]])
    for score in scores:
        print(f"Score: {score:.4f}")
    
    score_a = model.predict([["My first", "sentence pair"]])
    score_b = model.predict([["Second text", "pair"]])
    print(f"Score A: {score_a[0]:.4f}")
    print(f"Score B: {score_b[0]:.4f}")

if __name__ == "__main__":
    main()