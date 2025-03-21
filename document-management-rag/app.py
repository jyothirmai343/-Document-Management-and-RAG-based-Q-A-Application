import os
import json
import io
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import asyncpg
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "database")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "tfidf_index")

class Document(BaseModel):
    id: Optional[int] = None
    title: str
    content: str
    metadata: Dict[str, Any] = {}

class Query(BaseModel):
    question: str
    
class Answer(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []

class ErrorResponse(BaseModel):
    detail: str

async def get_db_pool():
    try:
        pool = await asyncpg.create_pool(
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            host=DB_HOST,
            port=DB_PORT,
            command_timeout=60
        )
        return pool
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return None

def parse_metadata(metadata):
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    elif isinstance(metadata, dict):
        return metadata
    else:
        return {}

def extract_text_from_file(file_content, file_name):
    file_ext = os.path.splitext(file_name.lower())[1]
    
    if file_ext == '.pdf':
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error extracting text: {str(e)}"
    
    elif file_ext in ['.txt', '.csv', '.md']:
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_content.decode('latin-1')
            except Exception as e:
                return f"Error decoding text: {str(e)}"
    
    else:
        return f"Unsupported file type: {file_ext}"

def split_text(text, chunk_size=1000, chunk_overlap=200):
    if not text:
        return []
        
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunk = text[i:i + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

class TfidfVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.doc_texts = []
        self.doc_metadatas = []
        self.matrix = None
        
    def add_texts(self, texts, metadatas=None):
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        self.documents.extend(list(zip(texts, metadatas)))
        self.doc_texts.extend(texts)
        self.doc_metadatas.extend(metadatas)
        
        self.matrix = self.vectorizer.fit_transform(self.doc_texts)
        
    def similarity_search(self, query, k=3):
        if not self.doc_texts:
            return []
            
        query_vec = self.vectorizer.transform([query])
        
        similarity_scores = cosine_similarity(query_vec, self.matrix)[0]
        
        top_indices = np.argsort(similarity_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc_text, doc_metadata = self.documents[idx]
                doc = type('Document', (), {
                    'page_content': doc_text,
                    'metadata': doc_metadata
                })
                results.append(doc)
        
        return results
        
    def as_retriever(self, search_type=None, search_kwargs=None):
        if not search_kwargs:
            search_kwargs = {}
        k = search_kwargs.get('k', 3)
            
        class SimpleRetriever:
            def __init__(self, vector_store, k):
                self.vector_store = vector_store
                self.k = k
                
            def get_relevant_documents(self, query):
                return self.vector_store.similarity_search(query, self.k)
                
        return SimpleRetriever(self, k)
        
    def save_local(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
        with open(f"{path}_docs.json", 'w') as f:
            json.dump([(text, meta) for text, meta in zip(self.doc_texts, self.doc_metadatas)], f)
        
    @classmethod
    def load_local(cls, path):
        vector_store = cls()
        try:
            if os.path.exists(f"{path}_docs.json"):
                with open(f"{path}_docs.json", 'r') as f:
                    docs_data = json.load(f)
                    texts = [item[0] for item in docs_data]
                    metadatas = [item[1] for item in docs_data]
                    vector_store.add_texts(texts, metadatas)
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return vector_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the application...")
    app.state.has_error = False
    
    try:
        app.state.db_pool = await get_db_pool()
        if app.state.db_pool:
            logger.info("Database connection established")
        else:
            logger.warning("Failed to establish database connection")
            app.state.has_error = True
    except Exception as e:
        logger.error(f"Database startup error: {e}")
        app.state.db_pool = None
        app.state.has_error = True
    
    app.state.vector_store = None
    
    if os.path.exists(f"{VECTOR_STORE_PATH}_docs.json"):
        try:
            app.state.vector_store = TfidfVectorStore.load_local(VECTOR_STORE_PATH)
        except Exception:
            pass
    else:
        app.state.vector_store = TfidfVectorStore()
        
    yield
    
    if hasattr(app.state, "db_pool") and app.state.db_pool:
        await app.state.db_pool.close()
    
    if hasattr(app.state, "vector_store") and app.state.vector_store:
        try:
            os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
            app.state.vector_store.save_local(VECTOR_STORE_PATH)
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

app = FastAPI(
    title="Document Management and RAG-based Q&A API",
    description="API for managing documents and performing question answering with TF-IDF based retrieval",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)

async def get_db():
    if app.state.db_pool is None:
        app.state.db_pool = await get_db_pool()
        
    if app.state.db_pool is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
        
    async with app.state.db_pool.acquire() as connection:
        yield connection

async def init_db(connection):
    try:
        await connection.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}'
            )
        ''')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database initialization error: {str(e)}")

@app.get("/", response_model=Dict[str, Any])
async def root():
    status = "healthy"
    components = {
        "database": "connected" if app.state.db_pool else "disconnected",
        "vector_store": "initialized" if app.state.vector_store else "not initialized"
    }
    
    if app.state.has_error:
        status = "degraded"
    
    return {
        "message": "Document Management and RAG-based Q&A Application (TF-IDF Edition)",
        "status": status,
        "components": components
    }

@app.post("/documents/", response_model=Document, responses={500: {"model": ErrorResponse}})
async def create_document(document: Document, connection=Depends(get_db)):
    try:
        await init_db(connection)
        
        result = await connection.fetchrow(
            "INSERT INTO documents (title, content, metadata) VALUES ($1, $2, $3) RETURNING id",
            document.title, document.content, json.dumps(document.metadata)
        )
        document.id = result["id"]
        
        if app.state.vector_store:
            chunks = split_text(document.content)
            metadatas = [{"source": document.title, "doc_id": document.id} for _ in chunks]
            app.state.vector_store.add_texts(chunks, metadatas=metadatas)
        else:
            app.state.vector_store = TfidfVectorStore()
            chunks = split_text(document.content)
            metadatas = [{"source": document.title, "doc_id": document.id} for _ in chunks]
            app.state.vector_store.add_texts(chunks, metadatas=metadatas)
        
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")

@app.get("/documents/", response_model=List[Document], responses={500: {"model": ErrorResponse}})
async def read_documents(connection=Depends(get_db)):
    try:
        await init_db(connection)
        
        rows = await connection.fetch("SELECT * FROM documents")
        documents = []
        for row in rows:
            metadata_dict = parse_metadata(row["metadata"])
            
            documents.append(Document(
                id=row["id"],
                title=row["title"],
                content=row["content"],
                metadata=metadata_dict
            ))
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read documents: {str(e)}")

@app.get("/documents/{document_id}", response_model=Document, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def read_document(document_id: int, connection=Depends(get_db)):
    try:
        row = await connection.fetchrow("SELECT * FROM documents WHERE id = $1", document_id)
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        metadata_dict = parse_metadata(row["metadata"])
        
        return Document(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            metadata=metadata_dict
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read document: {str(e)}")

@app.post("/upload/", response_model=Document, responses={500: {"model": ErrorResponse}})
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    connection=Depends(get_db)
):
    try:
        await init_db(connection)
        
        content = await file.read()
        
        content_text = extract_text_from_file(content, file.filename)
        
        file_metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content)
        }
        
        result = await connection.fetchrow(
            "INSERT INTO documents (title, content, metadata) VALUES ($1, $2, $3) RETURNING id",
            title, content_text, json.dumps(file_metadata)
        )
        
        document = Document(
            id=result["id"],
            title=title,
            content=content_text,
            metadata=file_metadata
        )
        
        if app.state.vector_store:
            chunks = split_text(content_text)
            metadatas = [{"source": title, "doc_id": document.id} for _ in chunks]
            app.state.vector_store.add_texts(chunks, metadatas=metadatas)
        else:
            app.state.vector_store = TfidfVectorStore()
            chunks = split_text(content_text)
            metadatas = [{"source": title, "doc_id": document.id} for _ in chunks]
            app.state.vector_store.add_texts(chunks, metadatas=metadatas)
        
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@app.post("/query/", response_model=Answer, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def query_documents(query: Query):
    try:
        if not app.state.vector_store:
            raise HTTPException(status_code=400, detail="No documents have been indexed yet")
        
        retriever = app.state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        docs = retriever.get_relevant_documents(query.question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        question_words = set(query.question.lower().split())
        relevant_sentences = []
        
        for doc in docs:
            sentences = doc.page_content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_words = set(sentence.lower().split())
                if len(question_words.intersection(sentence_words)) > 0:
                    relevant_sentences.append(sentence)
        
        if relevant_sentences:
            answer_text = "Based on the documents:\n\n"
            answer_text += "\n".join(["- " + sentence for sentence in relevant_sentences[:5]])
            answer_text += "\n\nThe information most relevant to your question is found in the sections above."
        else:
            answer_text = "I couldn't find specific information directly addressing your question in the documents. Here's the most relevant content I found:\n\n"
            answer_text += context[:500] + "..."
        
        sources = []
        for doc in docs:
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return Answer(answer=answer_text, sources=sources)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.delete("/documents/{document_id}", response_model=Dict[str, str], responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def delete_document(document_id: int, connection=Depends(get_db)):
    try:
        row = await connection.fetchrow("SELECT id FROM documents WHERE id = $1", document_id)
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        await connection.execute("DELETE FROM documents WHERE id = $1", document_id)
        
        return {"message": f"Document {document_id} deleted successfully (note: chunks still in vector store)"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    status = "healthy"
    components = {
        "database": "connected" if app.state.db_pool else "disconnected",
        "vector_store": "initialized" if app.state.vector_store else "not initialized",
    }
    
    if app.state.has_error or not app.state.db_pool:
        status = "degraded"
    
    return {
        "status": status,
        "components": components
    }

@app.get("/models", response_model=Dict[str, Any])
async def available_models():
    return {
        "retrieval_method": "TF-IDF",
        "vector_store_path": VECTOR_STORE_PATH,
        "recommended_future_models": [
            {"id": "mistralai/Mistral-7B-Instruct-v0.2", "description": "Good general purpose model"},
            {"id": "sentence-transformers/all-mpnet-base-v2", "description": "Strong embedding model for retrieval"},
            {"id": "facebook/bart-large-cnn", "description": "Optimized for summarization"},
            {"id": "deepset/roberta-base-squad2", "description": "Specialized for QA"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=True
    )
