import os
from typing import List, Optional
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import json


class PDFQuestionAnswering:
    def __init__(self, pdf_path: str, chunk_size: int = 1000):
        """
        Initialize the PDF Question Answering system.

        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Size of text chunks for processing
        """
        # Load environment variables from .env file
        load_dotenv()

        # Get API key
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # API endpoint for Gemini 2.0 Flash
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load and process the PDF
        self.text_chunks = self._process_pdf(pdf_path, chunk_size)
        self.embeddings = self._create_embeddings()

    def _process_pdf(self, pdf_path: str, chunk_size: int) -> List[str]:
        """
        Process the PDF file and split it into chunks.

        Args:
            pdf_path (str): Path to the PDF file
            chunk_size (int): Size of text chunks

        Returns:
            List[str]: List of text chunks
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all text chunks.

        Returns:
            np.ndarray: Array of embeddings
        """
        return self.embedding_model.encode(self.text_chunks)

    def _get_relevant_chunks(self, question: str, top_k: int = 3) -> List[str]:
        """
        Get the most relevant text chunks for a given question.

        Args:
            question (str): The question to find relevant chunks for
            top_k (int): Number of chunks to return

        Returns:
            List[str]: List of most relevant text chunks
        """
        question_embedding = self.embedding_model.encode([question])[0]
        similarities = np.dot(self.embeddings, question_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.text_chunks[i] for i in top_indices]

    def ask_question(self, question: str) -> str:
        """
        Ask a question about the PDF content.

        Args:
            question (str): The question to ask

        Returns:
            str: The answer from the model
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        # Get relevant chunks
        relevant_chunks = self._get_relevant_chunks(question)

        # Construct the prompt
        prompt = f"""You are a helpful AI assistant that provides accurate and concise answers based on the given context.
        Based on the following context from a PDF document, please answer the question.
        If the answer cannot be found in the context, say "I cannot find the answer in the provided context."
        Keep your response focused and relevant to the question.

        Context:
        {' '.join(relevant_chunks)}

        Question: {question}
        """

        # Prepare the request payload
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        # Make API request
        request_url = f"{self.api_url}?key={self.api_key}"

        response = requests.post(
            request_url, headers={"Content-Type": "application/json"}, json=payload
        )

        # Check for errors
        if response.status_code != 200:
            error_msg = response.json().get("error", {}).get("message", "Unknown error")
            raise Exception(f"API request failed: {error_msg}")

        # Extract and return the response text
        response_json = response.json()
        try:
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected API response format: {str(e)}")


def main():
    # Example usage
    load_dotenv()

    if os.getenv("GEMINI_API_KEY") is None:
        print("Please set GEMINI_API_KEY environment variable")
        return

    pdf_path = os.path.join("test_manuals", "sagemaker-unified-studio-user-guide.pdf")
    qa_system = PDFQuestionAnswering(pdf_path)

    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break

        try:
            print("\nThinking...")
            answer = qa_system.ask_question(question)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
