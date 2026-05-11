import time
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

from agent_graph import _build_graph

langfuse_handler = CallbackHandler()

agent = _build_graph().compile().with_config({"recursion_limit": 10})


async def stream_agent_response(message: str):
    t0 = time.time()
    config = {"callbacks": [langfuse_handler]}
    first_token = True
    async for msg, metadata in agent.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="messages",
    ):
        if msg.content and metadata["langgraph_node"] == "agent":
            if first_token:
                print(f"[timing] first token: {time.time() - t0:.2f}s")
                first_token = False
            yield msg.content
    print(f"[timing] total: {time.time() - t0:.2f}s")
