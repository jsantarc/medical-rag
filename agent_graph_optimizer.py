"""
agent_graph.py — ReAct agent graph with query optimizer node.
"""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

from deps import make_obj
from tool import document_search
from query_optimizer import get_improved_query


AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DiabeticAssist, a clinical reference chatbot specializing in diabetes care.
You answer questions using verified medical documents only.
Always cite the source document when referencing specific facts.
Never provide personal medical advice or recommend starting/stopping treatments.
Add a safety disclaimer when discussing dosages, insulin regimens, or drug interactions.
Focus on topics including: Type 1 and Type 2 diabetes, prediabetes, gestational diabetes,
blood glucose management, HbA1c targets, medications (metformin, insulin, GLP-1 agonists, SGLT2 inhibitors),
dietary guidance, and diabetes complications.
If a question is outside the scope of diabetes care, say so clearly and do not call any tool.
You MUST call document_search before answering any medical question. Never answer from memory alone.
Call at most one tool per question."""),
    MessagesPlaceholder("messages"),
])

tools = [document_search]
tool_node = ToolNode(tools)
llm = AGENT_PROMPT | make_obj().bind_tools(tools)


async def call_model(state: MessagesState):
    response = await llm.ainvoke({"messages": state["messages"]})
    return {"messages": [response]}


async def optimizer_node(state: MessagesState):
    query = state["messages"][0].content
    answer = state["messages"][-1].content
    improved = await get_improved_query(query, answer)
    return {"messages": [HumanMessage(content=improved)]}


def should_continue(state: MessagesState):
    last = state["messages"][-1]

    if last.tool_calls:
        return "tools"

    human_messages = [m for m in state["messages"] if m.type == "human"]
    if len(human_messages) == 1:
        return "optimizer"

    return END


def _build_graph():
    graph = StateGraph(MessagesState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("optimizer", optimizer_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "optimizer": "optimizer", END: END}
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("optimizer", "agent")

    return graph