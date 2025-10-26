import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def augmented_generation(query: str, docs: list[dict]) -> str:
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {docs}

            Provide a comprehensive answer that addresses the query:"""
    
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def augmented_summarization(query: str, docs: list[dict]) -> str:
    prompt = f"""
                Provide information useful to this query by synthesizing information from multiple search results in detail.
                The goal is to provide comprehensive information so that users know what their options are.
                Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
                This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                Query: {query}
                Search Results:
                {docs}
                Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
                """
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def augmented_citations(query: str, docs: list[dict]) -> str:
    prompt = prompt = f"""Answer the question or provide information based on the provided documents.

                        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                        If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

                        Query: {query}

                        Documents:
                        {docs}

                        Instructions:
                        - Provide a comprehensive answer that addresses the query
                        - Cite sources using [1], [2], etc. format when referencing information
                        - If sources disagree, mention the different viewpoints
                        - If the answer isn't in the documents, say "I don't have enough information"
                        - Be direct and informative

                        Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def augmented_question(query: str, docs: list[dict]) -> str:
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                Question: {query}

                Documents:
                {docs}

                Instructions:
                - Answer questions directly and concisely
                - Be casual and conversational
                - Don't be cringe or hype-y
                - Talk like a normal person would in a chat conversation

                Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text