from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,SystemMessage

load_dotenv()



# lllm load
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

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
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()
tools=[search_tool,calculator,get_stock_price]
#binding the tools with llm


llm_with_tools=llm.bind_tools(tools)
#--------------
#state 
#--------------

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    
def chat_node(state:ChatState):
    messages=state["messages"]
    response=llm_with_tools.invoke(messages)
    return {"messages":[response]}

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
    
    
    
    
    