import time
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

from agent_graph import _build_graph

agent = _build_graph().compile().with_config({"recursion_limit": 10})


async def stream_agent_response(message: str):
    t0 = time.time()
    config = {"callbacks": [CallbackHandler()]}
    first_token = True
    routed = False
    async for msg, metadata in agent.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="messages",
    ):
        if not routed and metadata["langgraph_node"] == "tools":
            print(f"[route] {msg.name}")
            routed = True
        if msg.content and metadata["langgraph_node"] == "agent":
            if not routed:
                print("[route] llm")
                routed = True
            if first_token:
                print(f"[timing] first token: {time.time() - t0:.2f}s")
                first_token = False
            yield msg.content
    print(f"[timing] total: {time.time() - t0:.2f}s")
