from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

from deps import make_obj
from tool import document_search


AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DiabeticAssist, a clinical reference chatbot specializing in diabetes care.

RETRIEVAL RULES:
- For new medical questions, call document_search before answering.
- For follow-up questions or clarifications about a previous answer already in this conversation, you may respond directly without calling a tool.
- Call at most one tool per turn.
- If document_search returns no relevant results, say so explicitly. Do NOT answer from memory.

FAITHFULNESS RULES:
- Answer ONLY using information from the retrieved documents.
- Do NOT add facts, dosages, or clinical details that are not explicitly stated in the retrieved text.
- If the document does not contain enough information to answer fully, say so clearly.
- Always quote or closely paraphrase the source — never infer or extrapolate.

SAFETY RULES:
- Never provide personal medical advice or recommend starting/stopping treatments.
- Always add a safety disclaimer when discussing dosages, insulin regimens, or drug interactions.
- If a question is outside the scope of diabetes care, say so clearly and do not call any tool.

SCOPE:
Type 1 and Type 2 diabetes, prediabetes, gestational diabetes, blood glucose management,
HbA1c targets, medications (metformin, insulin, GLP-1 agonists, SGLT2 inhibitors),
dietary guidance, and diabetes complications."""),
    MessagesPlaceholder("messages"),
])

tools = [document_search]
tool_node = ToolNode(tools)
llm = AGENT_PROMPT | make_obj().bind_tools(tools)


async def call_model(state: MessagesState):
    response = await llm.ainvoke({"messages": state["messages"]})
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
