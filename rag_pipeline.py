from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "pdf_chat"
QDRANT_URL = "http://localhost:6333"


llm = ChatOpenAI(
    model="upstage/solar-pro-3:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Answer questions based ONLY on the context below.
If the answer isn't in the context, say "I don't know based on this document."

Context:
{context}""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


session_histories: dict = {}


def chat(session_id: str, question: str) -> str:
    if session_id not in session_histories:
        session_histories[session_id] = []

    chat_history = session_histories[session_id]

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # lcel
    chain = prompt | llm
    response = chain.invoke(
        {
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }
    )

    answer = response.content

    # history updation
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer


def clear_session(session_id: str):
    if session_id in session_histories:
        del session_histories[session_id]
