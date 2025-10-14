import string
import json
from nltk.stem import PorterStemmer

class TextUtils:
    def __init__(self, stopwords_path="data/stopwords.txt"):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.stemmer = PorterStemmer()

    def load_stopwords(self, path: str) -> list[str]:
        try:
            with open(path, "r") as f:
                return [word.strip().lower() for word in f]
        except FileNotFoundError:
            print(f"Error: Stopwords file '{path}' not found.")
            return list()
        except Exception as e:
            print(f"Error: An error occurred while loading stopwords: {e}")
            return list()

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [token for token in tokens if token not in self.stopwords]

    def tokenize(self, text: str) -> list[str]:
        tokens = self.remove_stopwords((self.remove_punctuation(text).lower().split()))
        # return [self.stemmer.stem(token) for token in tokens]
        #tokens = self.remove_punctuation(text).lower().split()
        return tokens

    def token_match(self, query: str, text: str) -> bool:
        query_tokens = self.tokenize(query)
        text_tokens = self.tokenize(text)
        return not query_tokens.isdisjoint(text_tokens)

    @staticmethod
    def load_movies(path="data/movies.json") -> dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: '{path}' file not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from '{path}'.")
            return {}