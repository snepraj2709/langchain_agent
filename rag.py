import os
import uuid
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

class RAGSystem:
    def __init__(self, pdf_path: str, use_local: bool = True):
        self.embeddings = OpenAIEmbeddings()
        self.pdf_path = pdf_path
        self.collection_name = "pdf_docs"
        self.embedding_dimension = 1536  # OpenAI embedding size

        # Local in-memory Qdrant (for tests)
        if use_local:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host="localhost", port=6333)

        self._load_and_store_pdf()

    def _load_and_store_pdf(self):
        print(f"Loading and chunking PDF: {self.pdf_path}")

        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

        chunks = [d.page_content for d in docs]
        print(f"Created {len(chunks)} text chunks.")

        # OpenAI Embedding Model
        vectors = self.embeddings.embed_documents(chunks)
        print(f"Generated {len(vectors)} embeddings.")

        # Create Qdrant collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dimension, distance=Distance.COSINE)
        )

        # Insert points
        points = []
        for i, vector in enumerate(vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": chunks[i]}
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Stored {len(points)} points in Qdrant.")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        print(f"Retrieving top {top_k} chunks for query: '{query}'")

        # Embed query using OpenAI model
        query_embedding = self.embeddings.embed_query(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )

        if not response.points:
            return ["No matching text found."]

        return [p.payload.get("text", "") for p in response.points]

    def get_collection_info(self) -> dict:
        count = self.client.count(
            collection_name=self.collection_name,
            exact=True
        ).count

        return {
            "vectors_count": count,
            "points_count": count,
            "status": "ready"
        }
