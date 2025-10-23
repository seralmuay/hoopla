import json
import numpy as np
from sentence_transformers import SentenceTransformer
from cli.lib.chunk_utils import semantic_chunk

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
            
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}
        all_chunks = []
        chunk_metadata = []
        for doc_id, doc in enumerate(documents):
            if doc['description'] is None or doc['description'].strip() == "":
                continue
            chunks = semantic_chunk(doc['description'], max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)
            for ichunk, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'movie_idx': doc['id'],
                    'chunk_idx': ichunk,
                    'total_chunks': total_chunks
                })
        
        # Encode all chunks at once
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        try:
            with open('cache/movie_chunk_embeddings.npy', 'wb') as f:
                np.save(f, self.chunk_embeddings)

            with open('cache/chunk_metadata.json', 'w') as f:
                json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save chunk embeddings or metadata to cache: {e}")

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}
        try:
            with open('cache/movie_chunk_embeddings.npy', 'rb') as f:
                self.chunk_embeddings = np.load(f)
            
            with open('cache/chunk_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata['chunks']
            print(f"Loaded {len(self.chunk_embeddings)} chunk embeddings and {len(self.chunk_metadata)} metadata entries from cache.")
        except FileNotFoundError:
            print("Chunk embeddings or metadata cache not found, building new chunk embeddings...")
            return self.build_chunk_embeddings(documents)
        except Exception as e:
            print("Failed to load chunk embeddings or metadata from cache, rebuilding...")
            return self.build_chunk_embeddings(documents)

        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        results = []
        movie_scores = {}
        #print(f"Searching {len(self.chunk_embeddings)} chunk embeddings for query: {query}")
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_emb)
            metadata = self.chunk_metadata[i]
            #print({'chunk_idx': metadata['chunk_idx'], "movie_idx": metadata['movie_idx'], 'score': score})
            movie_idx = self.documents.index(next(doc for doc in self.documents if doc['id'] == metadata['movie_idx']))
            #print(f"Movie index in documents: {movie_idx}")
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
            results.append({'chunk_idx': metadata['chunk_idx'], "movie_idx": movie_idx, 'score': score})
            # movie_idx: The index of the document in self.documents (you'll need to use self.chunk_metadata to map back to this)
        
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_movies = sorted_movie_scores[:limit]
        final_results = []
        for movie_idx, score in top_movies:
            document = self.documents[movie_idx]
            final_results.append(format_search_result(score, document))
        
        return final_results
    
def format_search_result(score, document):
    return {
        'score': score,
        'title': document['title'],
        'description': document['description'][:100],
        'id': document['id']
    }


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
