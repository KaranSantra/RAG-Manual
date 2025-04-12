import os
from typing import List
import requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np


class PDFQuestionAnswering:
    def __init__(self, pdf_path: str, chunk_size: int = 1000):
        """Initialize the PDF Question Answering system."""
        load_dotenv()

        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Process PDF and create embeddings
        self.text_chunks = self._process_pdf(pdf_path, chunk_size)
        self.embeddings = self.embedding_model.encode(self.text_chunks)

    def _process_pdf(self, pdf_path: str, chunk_size: int) -> List[str]:
        """Process the PDF file and split it into chunks."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Extract text from PDF
        pdf_reader = PdfReader(pdf_path)
        text = " ".join(page.extract_text() for page in pdf_reader.pages)

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

    def _get_relevant_chunks(self, question: str, top_k: int = 3) -> List[str]:
        """Get the most relevant text chunks for a given question."""
        question_embedding = self.embedding_model.encode([question])[0]
        similarities = np.dot(self.embeddings, question_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.text_chunks[i] for i in top_indices]

    def ask_question(self, question: str) -> str:
        """Ask a question about the PDF content."""
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

        # Make API request
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(
            f"{self.api_url}?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
        )

        # Check for errors
        if response.status_code != 200:
            error_msg = response.json().get("error", {}).get("message", "Unknown error")
            raise Exception(f"API request failed: {error_msg}")

        # Extract and return the response text
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected API response format: {str(e)}")


def main():
    """Main function to run the PDF QA system."""
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
