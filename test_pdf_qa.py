import unittest
import os
from pdf_qa import PDFQuestionAnswering


class TestPDFQuestionAnswering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up mock API key for testing
        os.environ["GEMINI_API_KEY"] = "test_key"

        # Initialize the QA system with the test PDF
        pdf_path = os.path.join(
            "test_manuals", "sagemaker-unified-studio-user-guide.pdf"
        )
        cls.qa_system = PDFQuestionAnswering(pdf_path)

    def test_pdf_loading(self):
        """Test if PDF is loaded correctly"""
        self.assertTrue(len(self.qa_system.text_chunks) > 0)
        self.assertIsNotNone(self.qa_system.embeddings)

    def test_question_answering(self):
        """Test if the system can answer a basic question about the content"""
        question = "What is Amazon SageMaker Studio?"
        response = self.qa_system.ask_question(question)

        # Check if response is not empty and is a string
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

        # Check if response contains relevant keywords
        self.assertTrue(
            any(
                keyword in response.lower()
                for keyword in ["sagemaker", "studio", "aws"]
            )
        )

    def test_invalid_question(self):
        """Test handling of empty or invalid questions"""
        with self.assertRaises(ValueError):
            self.qa_system.ask_question("")

        with self.assertRaises(ValueError):
            self.qa_system.ask_question(None)

    @classmethod
    def tearDownClass(cls):
        # Clean up environment variables
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]


if __name__ == "__main__":
    unittest.main()
