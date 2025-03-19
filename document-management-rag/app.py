# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import logging
from datetime import datetime
import uuid
import sqlite3
import numpy as np
import json
import shutil
from functools import lru_cache
import pickle

# For demonstration purposes, use simple libraries instead of requiring Ollama and PGVector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Document Management and RAG-based Q&A API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("db", exist_ok=True)

# Database initialization
DB_PATH = "db/docmanagement.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create documents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        title TEXT,
        description TEXT,
        file_path TEXT,
        file_type TEXT,
        user_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed BOOLEAN DEFAULT 0,
        collection_name TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# Path to store embeddings
EMBEDDINGS_DIR = "db/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Pydantic models for request/response
class UserCreate(BaseModel):
    username: str
    email: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime

class DocumentCreate(BaseModel):
    title: str
    description: Optional[str] = None

class DocumentResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    file_type: str
    created_at: str
    processed: bool

class QuestionRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

# Helper functions
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@lru_cache(maxsize=10)
def get_vectorizer():
    return TfidfVectorizer(stop_words='english')

# Function to extract text from various file types
def extract_text_from_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    elif file_ext == '.csv':
        text = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                text += " ".join(row) + "\n"
        return text
    
    else:  # Default to treating as text file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

# Function to chunk text into smaller segments
def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    
    return chunks

# Function to process document and generate embeddings
async def process_document(document_id: str):
    try:
        conn = get_db_connection()
        document = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not document:
            logger.error(f"Document not found: {document_id}")
            conn.close()
            return
        
        document = dict(document)
        logger.info(f"Processing document: {document['title']}")
        
        # Extract text from document
        text = extract_text_from_file(document['file_path'])
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Generate collection name
        collection_name = f"doc_{document_id.replace('-', '_')}"
        
        # Generate embeddings using TF-IDF
        vectorizer = get_vectorizer()
        tfidf_matrix = vectorizer.fit_transform(chunks)
        
        # Save vectorizer, matrix, and chunks
        embedding_path = os.path.join(EMBEDDINGS_DIR, f"{collection_name}.pkl")
        with open(embedding_path, 'wb') as f:
            pickle.dump({
                'vectorizer': vectorizer,
                'tfidf_matrix': tfidf_matrix,
                'chunks': chunks
            }, f)
        
        # Update document in database
        conn.execute(
            "UPDATE documents SET processed = 1, collection_name = ? WHERE id = ?",
            (collection_name, document_id)
        )
        conn.commit()
        conn.close()
        
        logger.info(f"Document processed successfully: {document['title']}")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        try:
            conn.rollback()
            conn.close()
        except:
            pass

# API endpoints
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    conn = get_db_connection()
    user_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    try:
        conn.execute(
            "INSERT INTO users (id, username, email, created_at) VALUES (?, ?, ?, ?)",
            (user_id, user.username, user.email, created_at)
        )
        conn.commit()
        
        user_data = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        
        return {
            "id": user_data["id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "created_at": datetime.fromisoformat(user_data["created_at"])
        }
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Error creating user: {str(e)}")

@app.get("/users/", response_model=List[UserResponse])
async def get_users(skip: int = 0, limit: int = 100):
    conn = get_db_connection()
    users = conn.execute("SELECT * FROM users LIMIT ? OFFSET ?", (limit, skip)).fetchall()
    conn.close()
    
    return [{
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "created_at": datetime.fromisoformat(user["created_at"])
    } for user in users]

@app.post("/documents/", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    # Check if user exists
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    # Create document directory if it doesn't exist
    upload_dir = os.path.join("uploads", user_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get file extension
    file_type = os.path.splitext(file.filename)[1].replace(".", "")
    
    # Create document in database
    document_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    conn.execute(
        """
        INSERT INTO documents 
        (id, title, description, file_path, file_type, user_id, created_at, processed) 
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (document_id, title, description, file_path, file_type, user_id, created_at)
    )
    conn.commit()
    
    document_data = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    conn.close()
    
    # Process document in background
    background_tasks.add_task(process_document, document_id)
    
    return {
        "id": document_data["id"],
        "title": document_data["title"],
        "description": document_data["description"],
        "file_type": document_data["file_type"],
        "created_at": document_data["created_at"],
        "processed": bool(document_data["processed"])
    }

@app.get("/documents/", response_model=List[DocumentResponse])
async def get_documents(user_id: Optional[str] = None, skip: int = 0, limit: int = 100):
    conn = get_db_connection()
    
    if user_id:
        documents = conn.execute(
            "SELECT * FROM documents WHERE user_id = ? LIMIT ? OFFSET ?",
            (user_id, limit, skip)
        ).fetchall()
    else:
        documents = conn.execute(
            "SELECT * FROM documents LIMIT ? OFFSET ?",
            (limit, skip)
        ).fetchall()
    
    conn.close()
    
    return [{
        "id": doc["id"],
        "title": doc["title"],
        "description": doc["description"],
        "file_type": doc["file_type"],
        "created_at": doc["created_at"],
        "processed": bool(doc["processed"])
    } for doc in documents]

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    conn = get_db_connection()
    document = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    conn.close()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "id": document["id"],
        "title": document["title"],
        "description": document["description"],
        "file_type": document["file_type"],
        "created_at": document["created_at"],
        "processed": bool(document["processed"])
    }

@app.post("/qa/", response_model=AnswerResponse)
async def question_answering(request: QuestionRequest):
    try:
        conn = get_db_connection()
        
        # Get document collections to search
        collections = []
        document_titles = []
        
        if request.document_ids and len(request.document_ids) > 0:
            for doc_id in request.document_ids:
                doc = conn.execute("SELECT * FROM documents WHERE id = ? AND processed = 1", (doc_id,)).fetchone()
                if doc and doc["collection_name"]:
                    collections.append(doc["collection_name"])
                    document_titles.append(doc["title"])
        else:
            # If no specific documents, search all processed documents
            docs = conn.execute("SELECT * FROM documents WHERE processed = 1").fetchall()
            for doc in docs:
                if doc["collection_name"]:
                    collections.append(doc["collection_name"])
                    document_titles.append(doc["title"])
        
        conn.close()
        
        if not collections:
            raise HTTPException(status_code=400, detail="No processed documents available")
        
        # Retrieve relevant chunks from all collections
        all_relevant_chunks = []
        all_similarity_scores = []
        
        for collection in collections:
            try:
                embedding_path = os.path.join(EMBEDDINGS_DIR, f"{collection}.pkl")
                if not os.path.exists(embedding_path):
                    continue
                    
                with open(embedding_path, 'rb') as f:
                    data = pickle.load(f)
                    
                vectorizer = data['vectorizer']
                tfidf_matrix = data['tfidf_matrix']
                chunks = data['chunks']
                
                # Transform the question
                question_vector = vectorizer.transform([request.question])
                
                # Calculate similarity
                similarity = cosine_similarity(question_vector, tfidf_matrix).flatten()
                
                # Get top 3 most similar chunks
                top_indices = similarity.argsort()[-3:][::-1]
                for idx in top_indices:
                    if similarity[idx] > 0.1:  # Threshold to filter irrelevant results
                        all_relevant_chunks.append(chunks[idx])
                        all_similarity_scores.append(similarity[idx])
            except Exception as e:
                logger.error(f"Error processing collection {collection}: {str(e)}")
        
        if not all_relevant_chunks:
            return AnswerResponse(
                question=request.question,
                answer="I couldn't find any relevant information to answer your question.",
                sources=[]
            )
        
        # Sort chunks by similarity score
        combined = list(zip(all_relevant_chunks, all_similarity_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5 chunks
        top_chunks = [chunk for chunk, _ in combined[:5]]
        context = "\n\n".join(top_chunks)
        
        # Simple answer generation (in production, this would use an LLM)
        answer = f"Based on the documents, I found information related to your question. Here is a relevant excerpt:\n\n{context[:500]}..."
        
        # In a real implementation with LLM, you would use something like:
        # answer = llm.generate(prompt=f"Context: {context}\n\nQuestion: {request.question}\n\nAnswer:")
        
        return AnswerResponse(
            question=request.question,
            answer=answer,
            sources=document_titles
        )
    except Exception as e:
        logger.error(f"Error in question answering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)