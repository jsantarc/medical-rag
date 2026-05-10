import time
import aiosqlite

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langfuse.langchain import CallbackHandler

from deps import make_obj
from tool import document_search

langfuse_handler = CallbackHandler()

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DiabeticAssist, a clinical reference chatbot specializing in diabetes care.
You answer questions using verified medical documents only.
Always cite the source document when referencing specific facts.
Never provide personal medical advice or recommend starting/stopping treatments.
Add a safety disclaimer when discussing dosages, insulin regimens, or drug interactions.
Focus on topics including: Type 1 and Type 2 diabetes, prediabetes, gestational diabetes,
blood glucose management, HbA1c targets, medications (metformin, insulin, GLP-1 agonists, SGLT2 inhibitors),
dietary guidance, and diabetes complications.
If a question is outside the scope of available documents, say so clearly."""),
    MessagesPlaceholder("messages"),
])

tools = [document_search]
tool_node = ToolNode(tools)
llm = AGENT_PROMPT | make_obj().bind_tools(tools)

_agent = None


def call_model(state: MessagesState):
    response = llm.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def _build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph

async def _get_agent():
    global _agent
    if _agent is None:
        conn = await aiosqlite.connect("memory.db")
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        _agent = _build_graph().compile(checkpointer=checkpointer)
    return _agent

def reset_agent():
    global _agent
    _agent = None

async def stream_agent_response(message: str, session_id: str = "default"):
    t0 = time.time()
    agent = await _get_agent()
    print(f"[timing] agent ready: {time.time() - t0:.2f}s")

    config = {"configurable": {"thread_id": session_id}, "callbacks": [langfuse_handler]}
    first_token = True
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                if first_token:
                    print(f"[timing] first token: {time.time() - t0:.2f}s")
                    first_token = False
                yield token
    print(f"[timing] total: {time.time() - t0:.2f}s")
