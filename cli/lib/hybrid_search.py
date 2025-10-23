import os
import json
import time
from dotenv import load_dotenv
from google import genai


from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5, candidate_multiplier=3):
        """
        Perform weighted hybrid search combining BM25 and semantic search.
        
        Args:
            query: Search query string
            alpha: Weight for BM25 scores (0-1), where (1-alpha) is weight for semantic scores
            limit: Number of results to return
            candidate_multiplier: Multiplier for candidate retrieval (default: 3)
        
        Returns:
            List of search results with hybrid scores
        """
        # Get BM25 and semantic results with candidate multiplier for better coverage
        bm25_results = self._bm25_search(query, limit * candidate_multiplier)
        semantic_results = self.semantic_search.search_chunks(query, limit * candidate_multiplier)
        
        # Normalize scores
        bm25_scores_list = [score for _, score in bm25_results]
        normalized_bm25_scores = self.normalize_scores(bm25_scores_list)
        
        semantic_scores_list = [res['score'] for res in semantic_results]
        normalized_semantic_scores = self.normalize_scores(semantic_scores_list)
        
        # Create document lookup for O(1) access
        doc_lookup = {doc['id']: doc for doc in self.documents}
        
        # Build unified document scores dictionary
        doc_scores = {}
        
        # Add BM25 results
        for i, (doc_id, _) in enumerate(bm25_results):
            doc_scores[doc_id] = {
                'bm25_score': normalized_bm25_scores[i],
                'semantic_score': 0.0,
                'document': self.idx.docmap.get(doc_id)
            }
        
        # Merge semantic results
        for i, semantic_result in enumerate(semantic_results):
            doc_id = semantic_result['id']
            if doc_id in doc_scores:
                doc_scores[doc_id]['semantic_score'] = normalized_semantic_scores[i]
            else:
                doc_scores[doc_id] = {
                    'bm25_score': 0.0,
                    'semantic_score': normalized_semantic_scores[i],
                    'document': doc_lookup.get(doc_id, semantic_result)
                }
        
        # Calculate hybrid scores and build final results
        final_results = [
            {
                'id': doc_id,
                'hybrid_score': self.hybrid_score(
                    scores_dict['bm25_score'],
                    scores_dict['semantic_score'],
                    alpha
                ),
                'bm25_score': scores_dict['bm25_score'],
                'semantic_score': scores_dict['semantic_score'],
                'document': scores_dict['document']
            }
            for doc_id, scores_dict in doc_scores.items()
        ]
        
        # Sort by hybrid score and return top results
        final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return final_results[:limit]
    
    @staticmethod
    def hybrid_score(bm25_score, semantic_score, alpha=0.5):
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def rrf_search(self, query, k, limit=10, candidate_multiplier=3):
        bm25_results = self._bm25_search(query, limit * candidate_multiplier)
        semantic_results = self.semantic_search.search_chunks(query, limit * candidate_multiplier)

        # Create document lookup for O(1) access
        doc_lookup = {doc['id']: doc for doc in self.documents}
        
        doc_scores = {}
        
        # Add BM25 results with ranks
        for rank, (doc_id, _) in enumerate(bm25_results):
            doc_scores[doc_id] = {
                'bm25_rank': rank + 1,  # 1-indexed rank
                'semantic_rank': None,
                'document': self.idx.docmap.get(doc_id),
                'rrf_score': self.rrf_score(rank + 1,k)
            }
        
        # Merge semantic results with ranks
        for rank, semantic_result in enumerate(semantic_results):
            doc_id = semantic_result['id']
            if doc_id in doc_scores:
                doc_scores[doc_id]['semantic_rank'] = rank + 1  # 1-indexed rank
                tmp_score = self.rrf_score(rank + 1,k)
                doc_scores[doc_id]['rrf_score'] += tmp_score
            else:
                doc_scores[doc_id] = {
                    'bm25_rank': None,
                    'semantic_rank': rank + 1,  # 1-indexed rank
                    'document': doc_lookup.get(doc_id, semantic_result),
                    'rrf_score': self.rrf_score(rank + 1,k)
                }
        
        return sorted(
            [
                {
                    'id': doc_id,
                    'rrf_score': scores_dict['rrf_score'],
                    'bm25_rank': scores_dict['bm25_rank'],
                    'semantic_rank': scores_dict['semantic_rank'],
                    'document': scores_dict['document']
                }
                for doc_id, scores_dict in doc_scores.items()
            ],
            key=lambda x: x['rrf_score'],
            reverse=True
        )[:limit]    

    @staticmethod
    def rrf_score(rank, k=60):
        return 1 / (k + rank)
    
    @staticmethod
    def normalize_scores(scores) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score == 0:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    @staticmethod
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
    
    
    @staticmethod
    def spell_enhance_query(query: str) -> str:
        # Placeholder for spell correction logic
        # In a real implementation, integrate with a spell correction library or API
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")

        client = genai.Client(api_key=api_key)

        system_prompt = "You are a helpful assistant that corrects spelling mistakes in user queries."
        user_prompt = f"""Fix any spelling errors in this movie search query. Only correct obvious typos. Don't change correctly spelled words. Query: "{query}" If no errors, return the original query. Corrected:"""

        response = client.models.generate_content(
            model='gemini-2.0-flash-001', contents=f"{system_prompt}\n\n{user_prompt}"
        )
        return response.text.strip()
    
    @staticmethod
    def rewrite_enhance_query(query: str) -> str:
        # Placeholder for query rewriting logic
        # In a real implementation, integrate with a query rewriting library or API
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")

        client = genai.Client(api_key=api_key)

        system_prompt = "You are a helpful assistant that rewrites user queries to improve search results."
        user_prompt = f"""Rewrite this movie search query to be more specific and searchable.
                            Original: "{query}"

                            Consider:
                            - Common movie knowledge (famous actors, popular films)
                            - Genre conventions (horror = scary, animation = cartoon)
                            - Keep it concise (under 10 words)
                            - It should be a google style search query that's very specific
                            - Don't use boolean logic

                            Examples:

                            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                            Rewritten query:"""

        response = client.models.generate_content(
            model='gemini-2.0-flash-001', contents=f"{system_prompt}\n\n{user_prompt}"
        )
        return response.text.strip()
    
    @staticmethod
    def expand_enhance_query(query: str) -> list[str]:
        # Placeholder for query expansion logic
        # In a real implementation, integrate with a query expansion library or API
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")

        user_prompt = f"""Expand this movie search query with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        This will be appended to the original query.

                        Examples:

                        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                        - "action movie with bear" -> "action thriller bear chase fight adventure"
                        - "comedy with bear" -> "comedy funny bear humor lighthearted"

                        Query: "{query}"
                        """

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model='gemini-2.0-flash-001', contents=f"{user_prompt}"
        )
        return response.text.strip()
    
    @staticmethod
    def llm_ranking(query,limit,results):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")

        client = genai.Client(api_key=api_key)

        for doc in results:
            system_prompt = "You are a helpful assistant that ranks movie search results based on relevance to the user's query."
            user_prompt = f"""Rate how well this movie matches the search query.

                            Query: "{query}"
                            Movie: {doc['document']['title']} - {doc['document']['description']}

                            Consider:
                            - Direct relevance to query
                            - User intent (what they're looking for)
                            - Content appropriateness

                            Rate 0-10 (10 = perfect match).
                            Give me ONLY the number in your response, no other text or explanation.

                            Score:"""

            response = client.models.generate_content(
                model='gemini-2.0-flash-001', contents=f"{system_prompt}\n\n{user_prompt}"
            )
            try:
                score = int(response.text.strip())
            except ValueError:
                score = 5  # Default to neutral if parsing fails
            
            doc['llm_score'] = score
            time.sleep(3)  # To avoid rate limiting
        
        return sorted(
            results,
            key=lambda x: x['llm_score'],
            reverse=True
        )[:limit]
    
    @staticmethod
    def llm_ranking_batch(query,limit,results):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        ids =  [item['id'] for item in results]
        print(f"Original IDs: {ids}")

        system_prompt = "You are a helpful assistant that ranks movie search results based on relevance to the user's query."
        user_prompt = f"""Rank these movies by relevance to the search query.

                        Query: "{query}"

                        Movies:
                        {chr(10).join([f"ID:{doc['id']} - {doc['document']['title']} - {doc['document']['description']}" for doc in results])}

                        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                        [75, 12, 34, 2, 1]
                        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-001', contents=f"{system_prompt}\n\n{user_prompt}"
        )
        
        try:
            # Strip markdown code fences if present
            cleaned_response = HybridSearch.strip_markdown_fences(response.text)
            ranks = json.loads(cleaned_response)
        except ValueError:
            ranks = [doc['id'] for doc in results]  # Default to original order if parsing fails
            
        order_map = {id_val: idx for idx, id_val in enumerate(ranks)}

        ordered_results = sorted(results, key=lambda x: order_map[x['id']])[:limit]

        return ordered_results