# PDF Question Answering System

This is a Python application that allows you to ask questions about PDF documents using Google's Gemini AI. The system uses RAG (Retrieval Augmented Generation) to provide accurate answers based on the content of your PDF files.

## Features

- PDF text extraction and chunking
- Semantic search using sentence transformers
- Question answering using Google's Gemini AI
- Interactive command-line interface

## Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini AI

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

1. Place your PDF file in the `test_manuals` directory
2. Run the script:
   ```bash
   python pdf_qa.py
   ```
3. Enter your questions when prompted
4. Type 'quit' to exit

## Running Tests

To run the test suite:

```bash
python -m unittest test_pdf_qa.py
```

## How it Works

1. The system first loads and processes the PDF file into manageable chunks
2. For each question:
   - Creates embeddings for the question
   - Finds the most relevant chunks using semantic similarity
   - Uses Gemini AI to generate an answer based on the relevant context

## Note

Make sure you have a valid Google API key with access to the Gemini API. You can obtain one from the Google AI Studio.
