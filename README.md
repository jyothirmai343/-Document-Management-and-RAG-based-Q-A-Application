# Document Management and RAG-based Q&A API

A FastAPI application to manage documents and answer questions on their content utilizing Retrieval-Augmented Generation (RAG).

## Overview

This project is a lightweight but efficient API for document management and question answering. It enables users to:

- Upload documents of different formats (PDF, CSV, TXT)
- Process documents to get text and vector embeddings
- Ask questions regarding document content
- Retrieve appropriate answers with source pointers

Rather than having to use heavyweight infrastructure such as Ollama and PGVector, this system employs lightweight counterparts (scikit-learn's TF-IDF Vectorizer) to produce embeddings and similarity scores.

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- SQLite
- scikit-learn
- PyPDF
- Other requirements listed in requirements.txt

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/document-qa-api.git
cd document-qa-api
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python app.py
```

The API will be accessible at `http://localhost:8000`.

## API Endpoints

### User Management

- `POST /users/` - Create a new user
- `GET /users/` - Get all users

### Document Management

- `POST /documents/` - Upload a document
- `GET /documents/` - Get all documents
- `GET /documents/{document_id}` - Get document details

### Question Answering

- `POST /qa/` - Ask a question about document content

## How It Works

1. **Document Processing:**
- Documents are uploaded and stored in the `uploads` directory
- Text is pulled out of various file formats
- Text is segmented into smaller units
- TF-IDF embeddings are created for every segment
- Embeddings are cached to be retrieved later

2. **Question Answering:**
- User provides a question and optional doc IDs
- System converts the question to a TF-IDF vector
- Similarity search identifies the most appropriate chunks
- Top chunks are merged to create a context
- (In production, an LLM would produce an answer based on this context)

## Future Improvements

- Incorporate authentication and authorization
- Handle more document formats
- Use a proper LLM for answer generation
- Incorporate document tagging and categorization
- Enhance chunking strategies for improved context retrieval
- Add a web interface for easier interaction
