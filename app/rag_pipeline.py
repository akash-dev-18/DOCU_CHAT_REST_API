import os
from typing import AsyncGenerator
from langchain_openai import ChatOpenAI

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient


COLLECTION_NAME = "pdf_chat"
QDRANT_URL = os.getenv("QDRANT_URL")

llm = ChatOpenAI(
    model="upstage/solar-pro-3:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
)

llm_stream = ChatOpenAI(
    model="upstage/solar-pro-3:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.2,
    streaming=True,
)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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


def get_retriever():
    client = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": 4})


async def chat(session_id: str, question: str) -> str:
    if session_id not in session_histories:
        session_histories[session_id] = []

    chat_history = session_histories[session_id]

    retriever = get_retriever()
    docs = await retriever.ainvoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm
    response = await chain.ainvoke(
        {
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }
    )

    answer = response.content
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer


async def chat_stream(session_id: str, question: str) -> AsyncGenerator[str, None]:
    if session_id not in session_histories:
        session_histories[session_id] = []

    chat_history = session_histories[session_id]

    retriever = get_retriever()
    docs = await retriever.ainvoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm_stream
    full_answer = ""

    async for chunk in chain.astream(
        {
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }
    ):
        token = chunk.content
        if token:
            full_answer += token
            yield token

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=full_answer))


def clear_session(session_id: str):
    if session_id in session_histories:
        del session_histories[session_id]
