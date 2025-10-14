import os
import pickle
from text_utils import TextUtils
from collections import Counter


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.text_utils = TextUtils()

    def __add_document(self, doc_id: int, text: str):
        tokens = self.text_utils.remove_stopwords(text.lower().split())
        cnt = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            cnt[token] += 1

        self.term_frequencies[doc_id] = cnt

    # def get_documents(self, term) -> list[dict]:
    #     doc_ids = self.index.get(term.lower(), set())
    #     return [self.docmap[doc_id] for doc_id in sorted(doc_ids)]

    def get_documents(self, term) -> list[dict]:
        term = term.lower()
        matched_doc_ids = set()
        for key in self.index:
            if key.startswith(term):
                matched_doc_ids.update(self.index[key])
        return [self.docmap[doc_id] for doc_id in sorted(matched_doc_ids)]
    
    # def get_tf(self, doc_id: int, term: str) -> int:
    #     term = term.lower()
    #     if doc_id in self.term_frequencies:
    #         return self.term_frequencies[doc_id].get(term, 0)
    #     return 0
    
    def get_tf(self, doc_id: int, term: str) -> int:
        term = term.lower()
        if doc_id in self.term_frequencies:
        # Sum counts for all tokens that start with the term
            return sum(
                count for token, count in self.term_frequencies[doc_id].items()
                if token.startswith(term)
            )
        return 0

    def build(self):
        movies = TextUtils.load_movies()
        for movie in movies["movies"]:
            doc_id = movie["id"]
            text = f"{self.text_utils.remove_punctuation(movie['title'])} {self.text_utils.remove_punctuation(movie['description'])}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump((self.index), f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        try:    
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)

            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Cache files not found. Please build the index first.")

