import time
import aiosqlite
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent_graph import _build_graph

_graph = _build_graph()
_checkpointer = None
_agent = None


async def get_agent():
    global _checkpointer, _agent
    if _agent is None:
        conn = await aiosqlite.connect("memory.db")
        _checkpointer = AsyncSqliteSaver(conn)
        await _checkpointer.setup()
        _agent = _graph.compile(checkpointer=_checkpointer).with_config({"recursion_limit": 5})
    return _agent


async def stream_agent_response(message: str, session_id: str = "default"):
    t0 = time.time()
    agent = await get_agent()
    config = {
        "callbacks": [CallbackHandler()],
        "configurable": {"thread_id": session_id},
    }
    first_token = True
    async for msg, metadata in agent.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="messages",
    ):
        if metadata["langgraph_node"] == "tools":
            print(f"[route] {msg.name}")
        if msg.content and metadata["langgraph_node"] == "agent":
            if first_token:
                print(f"[timing] first token: {time.time() - t0:.2f}s")
                first_token = False
            yield msg.content
    print(f"[timing] total: {time.time() - t0:.2f}s")
