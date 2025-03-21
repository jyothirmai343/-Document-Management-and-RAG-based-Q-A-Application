
# Document Management and RAG API Documentation

## Overview

This is a FastAPI application that provides document management and retrieval-augmented generation (RAG) capabilities using TF-IDF for similarity search. The application allows users to upload documents, store them in a PostgreSQL database, and query the documents using natural language questions.

## Features

- Document management (create, read, delete)
- File upload and text extraction from multiple formats (PDF, TXT, CSV, MD)
- Text chunking for efficient storage and retrieval
- TF-IDF based similarity search
- Simple question answering capabilities
- Health monitoring endpoints

## Technical Stack

- **Web Framework**: FastAPI
- **Database**: PostgreSQL (via asyncpg)
- **Vector Store**: Custom TF-IDF implementation
- **Text Processing**: scikit-learn for TF-IDF vectorization
- **File Handling**: PyPDF2 for PDF processing

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with system status |
| `/documents/` | POST | Create a new document |
| `/documents/` | GET | List all documents |
| `/documents/{document_id}` | GET | Get a specific document |
| `/upload/` | POST | Upload a file as a document |
| `/query/` | POST | Query the documents |
| `/documents/{document_id}` | DELETE | Delete a document |
| `/health` | GET | Health check endpoint |
| `/models` | GET | Information about available models |

### Detailed Endpoint Descriptions

#### Root Endpoint (`GET /`)

Returns the status of the application and its components.

**Response**:
```json
{
  "message": "Document Management and RAG-based Q&A Application (TF-IDF Edition)",
  "status": "healthy",
  "components": {
    "database": "connected",
    "vector_store": "initialized"
  }
}
```

#### Create Document (`POST /documents/`)

Creates a new document in the database and adds it to the vector store.

**Request Body**:
```json
{
  "title": "Example Document",
  "content": "This is the document content.",
  "metadata": {
    "author": "John Doe",
    "date": "2025-03-21"
  }
}
```

**Response**:
```json
{
  "id": 1,
  "title": "Example Document",
  "content": "This is the document content.",
  "metadata": {
    "author": "John Doe",
    "date": "2025-03-21"
  }
}
```

#### List Documents (`GET /documents/`)

Returns a list of all documents in the database.

**Response**:
```json
[
  {
    "id": 1,
    "title": "Example Document",
    "content": "This is the document content.",
    "metadata": {
      "author": "John Doe",
      "date": "2025-03-21"
    }
  },
  {
    "id": 2,
    "title": "Another Document",
    "content": "More content here.",
    "metadata": {
      "author": "Jane Smith",
      "date": "2025-03-20"
    }
  }
]
```

#### Get Document (`GET /documents/{document_id}`)

Returns a specific document by ID.

**Response**:
```json
{
  "id": 1,
  "title": "Example Document",
  "content": "This is the document content.",
  "metadata": {
    "author": "John Doe",
    "date": "2025-03-21"
  }
}
```

#### Upload Document (`POST /upload/`)

Uploads a file as a document. Uses form data with a file upload and title field.

**Form Fields**:
- `file`: The file to upload
- `title`: The title for the document

**Response**: Same as Create Document

#### Query Documents (`POST /query/`)

Queries the documents using natural language and returns relevant information.

**Request Body**:
```json
{
  "question": "What is the main topic of the example document?"
}
```

**Response**:
```json
{
  "answer": "Based on the documents:\n\n- This is the document content.\n\nThe information most relevant to your question is found in the sections above.",
  "sources": [
    {
      "content": "This is the document content.",
      "metadata": {
        "source": "Example Document",
        "doc_id": 1
      }
    }
  ]
}
```

#### Delete Document (`DELETE /documents/{document_id}`)

Deletes a document from the database.

**Response**:
```json
{
  "message": "Document 1 deleted successfully (note: chunks still in vector store)"
}
```

#### Health Check (`GET /health`)

Returns the health status of the application.

**Response**:
```json
{
  "status": "healthy",
  "components": {
    "database": "connected",
    "vector_store": "initialized"
  }
}
```

#### Available Models (`GET /models`)

Returns information about the retrieval method and recommended models.

**Response**:
```json
{
  "retrieval_method": "TF-IDF",
  "vector_store_path": "tfidf_index",
  "recommended_future_models": [
    {
      "id": "mistralai/Mistral-7B-Instruct-v0.2",
      "description": "Good general purpose model"
    },
    {
      "id": "sentence-transformers/all-mpnet-base-v2",
      "description": "Strong embedding model for retrieval"
    },
    {
      "id": "facebook/bart-large-cnn",
      "description": "Optimized for summarization"
    },
    {
      "id": "deepset/roberta-base-squad2",
      "description": "Specialized for QA"
    }
  ]
}
```

## Core Components

### Document Model

The `Document` class represents a document in the system with the following fields:
- `id`: Optional integer, the document's ID in the database
- `title`: String, the document's title
- `content`: String, the document's content
- `metadata`: Dictionary, additional information about the document

### TF-IDF Vector Store

The `TfidfVectorStore` class provides vector storage and similarity search using TF-IDF:
- `add_texts`: Adds text chunks to the store
- `similarity_search`: Finds similar documents to a query
- `as_retriever`: Returns a retriever object for the vector store
- `save_local` / `load_local`: Persist and load the vector store

### Text Processing

- `extract_text_from_file`: Extracts text from different file formats
- `split_text`: Splits text into chunks with overlap
- `parse_metadata`: Ensures metadata is properly formatted

## Configuration

The application uses environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_USER` | PostgreSQL username | `postgres` |
| `DB_PASSWORD` | PostgreSQL password | `password` |
| `DB_NAME` | PostgreSQL database name | `database` |
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `VECTOR_STORE_PATH` | Path to store vector indices | `tfidf_index` |

## Running the Application

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Required Python packages (see import statements)

### Starting the Server

The application can be run with uvicorn:

```bash
python app.py
```

This starts the server on `0.0.0.0:8000` with auto-reload enabled.

### Docker (Optional)

A Dockerfile could be created with the following content:

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Limitations and Future Improvements

1. The current implementation does not support updating documents
2. When documents are deleted from the database, their chunks remain in the vector store
3. The TF-IDF implementation is basic and could be enhanced with more sophisticated techniques
4. The query response is simplistic and could be improved with a more advanced LLM integration
5. Authentication and authorization are not implemented
6. No pagination for document listing
