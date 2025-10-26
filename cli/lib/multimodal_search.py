from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import load_movies, cosine_similarity


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents: list[dict] = None):
        self.model = SentenceTransformer("clip-ViT-B-32")
        self.documents = documents
        self.texts = []
        self.text_embeddings = None
        #self.text_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_image(self, image_path: str) -> list[dict]:
        #load the image
        image = Image.open(image_path)
        #convert the image to a tensor
        image_embedding = self.model.encode(image)
        #return the image
        return image_embedding

    def embed_text(self, documents: list[dict]):
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def search_with_image(self, image_path: str) -> list[dict]:
        image_embedding = self.embed_image(image_path)

        similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:5]:
            results.append({
                "id": doc["id"],
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            })

        return results

def verify_image_embeddings(image_path: str):
    multimodal_search = MultimodalSearch()
    image_embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    documents = load_movies()
    multimodal_search = MultimodalSearch(documents=documents)
    multimodal_search.embed_text(documents)

    results = multimodal_search.search_with_image(image_path)

    return results