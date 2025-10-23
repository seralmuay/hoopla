import json
import os
from time import sleep

from dotenv import load_dotenv
from google import genai

from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {doc.get("title", "")} - {doc.get("document", "")}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Give me ONLY the number in your response, no other text or explanation.

                    Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def llm_rerank_batch(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    if not documents:
        return []

    doc_map = {}
    doc_list = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"ID: {doc_id} - {doc.get('title', '')} - {doc.get('document', '')}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank these movies by relevance to the search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                [75, 12, 34, 2, 1]
                """

    response = client.models.generate_content(model=model, contents=prompt)
    ranking_text = strip_markdown_fences(response.text)

    parsed_ids = json.loads(ranking_text)

    reranked = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})

    return reranked[:limit]


def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    if method == "batch":
        return llm_rerank_batch(query, documents, limit)
    if method == "cross_encoder":
        return llm_rerank_cross_encoder(query, documents, limit)
    else:
        return documents[:limit]
    
def llm_rerank_cross_encoder(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    for doc in documents:
        #pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
        score =cross_encoder.predict([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
        doc["cross_encoder_score"]=score
    
    documents.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return documents[:limit]

    
def strip_markdown_fences(text: str) -> str:
        """Remove markdown code fences from LLM response if present."""
        text = text.strip()
        # Check if wrapped in markdown code fences
        if text.startswith('```'):
            # Remove opening fence (```json or ````
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)
        return text.strip()