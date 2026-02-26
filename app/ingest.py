from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings


load_dotenv()

COLLECTION_NAME = "pdf_chat"
QDRANT_URL = os.getenv("QDRANT_URL")


async def ingest_pdf(pdf_path: str):
    print("Loading pdf")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )

    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(  # 👈 no await needed
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    await vector_store.aadd_documents(chunks)
    print("FILE INDEXED")

    return {"chunks_created": len(chunks), "status": "success"}
