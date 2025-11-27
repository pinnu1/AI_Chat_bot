import os
import tempfile
from langgraph.graph import StateGraph,START,END
from typing import Dict, Optional, TypedDict, Annotated,Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings



from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage

load_dotenv()
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}



# lllm load
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    transport="rest"
)


#tool to search the web pages 

search_tool=DuckDuckGoSearchRun(region="us-en")


#customilize toool 
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    
def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None
 
def ingest_file(file_bytes: bytes, thread_id: str,filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("File bytes are required for ingestion.")
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or "uploaded_file.pdf",
            "num_pages": len(docs),
            "num_chunks": len(chunks),
        }
        return {
            "filename": filename or "uploaded_file.pdf",
            "num_pages": len(docs),
            "num_chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }
tools=[search_tool,calculator,get_stock_price, rag_tool]
#binding the tools with llm


llm_with_tools=llm.bind_tools(tools)
#--------------
#state 
#--------------

class ChatState(TypedDict):
    #add_messages is same as operator add to add the diffrent message in the list so that previous message will not be lost
    messages:Annotated[list[BaseMessage],add_messages]

#-------------
#chat node function how chatting will happen 
#-----------

def chat_node(state:ChatState,config=None):
    thread_id=None
    if config and isinstance(config,dict):
        thread_id=config.get("configurable",{}).get("thread_id")
    system_message=SystemMessage(
        content=(
             "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
        
        
    )
    messages=[system_message,*state["messages"]]
    response=llm_with_tools.invoke(messages,config=config)
    return {"messages":[response]}

#-----------------
#tool node langgraph prebuilt node to handle the tools
#-----------------


tool_node=ToolNode(tools)

#-----------------
#chekpointer 
#---------------------


conn = sqlite3.connect('chat_history.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
#---------------
#graph
#---------------

graph=StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)



graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


  
# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads) 
 
def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
    
    
    
    
    