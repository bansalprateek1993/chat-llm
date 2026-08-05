from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import requests
import math
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage


load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)


def get_retreiver():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    retreiver = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":4}
    )
    return retreiver

@tool
def rag_tool(query:str)->str:
    """
    Retreive relevant information from the pdf document.
    Use this tool when the user asks the factual or conceptual questions
    that may be answered using the stored pdf documents.
    Args:
        query:The question or search query used to retreive the pdf content.
    """
    retreiver = get_retreiver()
    documents = retreiver.invoke(query)
    if not documents:
        return "No releavnt information was found in the pdf"

    formattted_docs = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unkown source")
        page = document.metadata.get("page", "Unkown page")

        formattted_docs.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )
    
    return "\n\n".join(formattted_docs)


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_KEY")
)

search_tool = TavilySearch(
    max_results=5,
    topic = "general",
    search_depth="advanced",
    tavily_api_key=os.getenv("TAVILY_API_KEY")  
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math caclulations.
    Input should be valid math expression.
    Example: 2+2, math.sqrt(16), 10*5
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }
        result = eval(expression, {"__builtin__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock proce for a given symbol (e.g. AAPL, TSLA)
    using Aplha vantage with API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=FGY6PWNXREZFY603"
    r = requests.get(url)
    return r.json()

# Make tool list
tools = [rag_tool, search_tool, calculator, get_stock_price]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    "LLM node that may answer directly or call an appropiate tool"

    system_message = SystemMessage(
    content="""
        You are an intelligent agentic assistant with access to external tools.

        # Primary Objective
        Provide accurate, concise, and helpful answers. Use tools whenever they can improve correctness, freshness, or completeness.

        # Available Tools

        1. rag_tool
        Purpose:
        - Answer questions about uploaded documents or PDFs.
        Rules:
        - Always retrieve relevant context before answering document-related questions.
        - Never fabricate information that is not present in the retrieved context.
        - If no document is available, ask the user to upload one.

        2. search_tool
        Purpose:
        - Retrieve recent, current, or internet-based information.
        Examples:
        - News
        - Current events
        - Recent company information
        - Latest technologies

        3. calculator
        Purpose:
        - Perform mathematical calculations.
        Rules:
        - Use this tool for arithmetic and multi-step calculations.
        - Do not perform complex calculations mentally.

        4. get_stock_price
        Purpose:
        - Retrieve the latest stock price information.

        # Decision Policy
        - For document questions, use rag_tool.
        - For current or internet-based information, use search_tool.
        - For calculations, use calculator.
        - For stock prices, use get_stock_price.
        - For general knowledge questions, answer directly without tools.

        # Response Guidelines
        - Do not hallucinate facts.
        - Prefer tool results over assumptions.
        - If a tool fails, explain the issue and suggest alternatives.
        - After receiving tool outputs, provide a clear, complete, and user-friendly response.
        """
        )
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    threads = checkpoint.list(None)
    all_threads = set()
    for thread in threads:
        all_threads.add(thread.config['configurable']['thread_id'])

    return list(all_threads)
# CONFIG = {"configurable":{"thread_id": "default_thread"}}
# res = chatbot.invoke(
#     {"messages": [HumanMessage(content="Hello, how is it going?")]},
#     config=CONFIG
# )

# print(res)