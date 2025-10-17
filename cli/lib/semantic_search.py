import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text is None or text.strip() == "":
            raise ValueError("Input text cannot be empty or None")
        
        input_text = [text]
        embedding = self.model.encode(input_text)
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}
        doc_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)

        try:
            with open('cache/movie_embeddings.npy', 'wb') as f:
                np.save(f, self.embeddings)
        except Exception as e:
            raise IOError(f"Failed to save embeddings to cache: {e}")
        
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}
        try:
            with open('cache/movie_embeddings.npy', 'rb') as f:
                self.embeddings = np.load(f)
        except FileNotFoundError:
            return self.build_embeddings(documents)
        except Exception as e:
            raise IOError(f"Failed to load embeddings from cache: {e}")
        
        return self.embeddings
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        results = []
        for i, doc_emb in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding,doc_emb)
            document = self.document_map[i]
            results.append((score, document))
        results.sort(key=lambda x: x[0], reverse=True)
        formatted_results = []
        for score, document in results[:limit]:
            formatted_results.append({
                'score': score,
                'title': document['title'],
                'description': document['description']
            })

        return formatted_results
            


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def load_documents():
    import json
    try:
        with open('data/movies.json', 'r') as f:
            data = json.load(f)
            return data.get("movies", [])
    except FileNotFoundError:
        raise FileNotFoundError("Movies data file not found.")
    except json.JSONDecodeError:
        raise ValueError("Error decoding the movies JSON file.")

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_documents()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    query_embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {query_embedding[:5]}")
    print(f"Shape: {query_embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)