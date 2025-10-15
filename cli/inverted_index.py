import os
import pickle
import math
from text_utils import TextUtils
from collections import Counter
from search_utils import BM25_K1, BM25_B


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.text_utils = TextUtils()
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = self.text_utils.tokenize(text)
        cnt = Counter()
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            cnt[token] += 1

        self.term_frequencies[doc_id] = cnt

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)

    def get_documents(self, term) -> list[dict]:
        term = term.lower()
        matched_doc_ids = set()
        for key in self.index:
            if key.startswith(term):
                matched_doc_ids.update(self.index[key])
        return [self.docmap[doc_id] for doc_id in sorted(matched_doc_ids)]
    
    def get_tf(self, doc_id: int, term: str) -> int:
        term = term.lower()
        if doc_id in self.term_frequencies:
            return self.term_frequencies[doc_id].get(self.text_utils.stem_text(term), 0)
        # Sum counts for all tokens that start with the term
            # return sum(
            #     count for token, count in self.term_frequencies[doc_id].items()
            #     if token.startswith(term)
            # )
        return 0
    
    def get_idf(self, term: str) -> float:
        term = term.lower()
        document_count = len(self.docmap)
        documents = len(self.index.get(self.text_utils.stem_text(term), set()))
        if documents == 0:
            return 0.0
        # Using smoothed IDF formula
        idf = math.log((document_count + 1) / (documents + 1))
        return idf
    
    def get_bm25_idf(self, term: str) -> float:
        terms = self.text_utils.tokenize(term)
        if len(terms) > 1:
            raise ValueError("BM25 IDF can only be calculated for a single term.")
        term = terms[0]
        
        document_count = len(self.docmap)
        documents = len(self.index.get(term, set()))
        if documents == 0:
            return 0.0
        idf = math.log((document_count - documents + 0.5) / (documents + 0.5) + 1)
        return idf

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0
        
        # Length normalization factor
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        # Using BM25 TF formula
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf
    
    def bm25(self, doc_id:int, term:str) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return tf * idf
    
    def bm25_search(self, query, limit):
        query_tokens = set(self.text_utils.tokenize(query))
        scores: dict[int, float] = {}

        for token in query_tokens:
            idf = self.get_bm25_idf(token)
            for doc_id in self.index.get(token, set()):
                tf = self.get_bm25_tf(doc_id, token)
                scores[doc_id] = scores.get(doc_id, 0.0) + tf * idf

        ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_docs = ranked_docs[:limit]
        return top_docs

    def build(self):
        movies = TextUtils.load_movies()
        for movie in movies["movies"]:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        try:
            os.makedirs("cache", exist_ok=True)

            with open("cache/index.pkl", "wb") as f:
                pickle.dump(self.index, f)

            with open("cache/docmap.pkl", "wb") as f:
                pickle.dump(self.docmap, f)

            with open("cache/term_frequencies.pkl", "wb") as f:
                pickle.dump(self.term_frequencies, f)

            with open("cache/doc_lengths.pkl", "wb") as f:
                pickle.dump(self.doc_lengths, f)

        except Exception as e:
            raise IOError(f"Error saving index to cache: {e}")

    def load(self):
        try:    
            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)

            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)

            with open("cache/doc_lengths.pkl", "rb") as f:
                self.doc_lengths = pickle.load(f)

        except FileNotFoundError:
            raise FileNotFoundError("Cache files not found. Please build the index first.")

