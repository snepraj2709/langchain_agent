from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, pdf_path: str):
        self.embeddings = OpenAIEmbeddings()
        self.client = QdrantClient(":memory:")
        self.collection_name = "pdf_docs"
        self.pdf_path = pdf_path
        self._setup_collection()
    
    def _setup_collection(self):
        """Load PDF and create embeddings"""
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        
        # Generate embeddings and store
        points = []
        for idx, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk.page_content)
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={"text": chunk.page_content}
                )
            )
        
        self.client.upsert(collection_name=self.collection_name, points=points)
    
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve relevant documents"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [hit.payload["text"] for hit in results]