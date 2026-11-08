"""Specialist agent as its own service: a LangGraph agent exposed over A2A (a2a-sdk 1.x, protocol 1.0).

Run:   python web_research_agent.py            (serves on $PORT, default 9001)
Card:  GET  http://localhost:9001/.well-known/agent-card.json
RPC:   POST http://localhost:9001/             (JSON-RPC: SendMessage / SendStreamingMessage / GetTask ...)

The web search tool is a MOCK (fixed corpus) so the prototype needs no search API key.
Set A2A_TOKEN to require "Authorization: Bearer <token>" on the RPC endpoint (the card stays public).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import llm, load_env  # noqa: E402

load_env()  # before building the model below

import uvicorn  # noqa: E402
from a2a.helpers.proto_helpers import new_task, new_text_message, new_text_part  # noqa: E402
from a2a.server.agent_execution import AgentExecutor, RequestContext  # noqa: E402
from a2a.server.events import EventQueue  # noqa: E402
from a2a.server.request_handlers import DefaultRequestHandler  # noqa: E402
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes  # noqa: E402
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater  # noqa: E402
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role, TaskState  # noqa: E402
from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

# ----------------------------------------------------------------------------- the LangGraph agent
NEWS = [
    "2026-08-20: The A2A (Agent2Agent) protocol moved to the Agentic AI Foundation, alongside MCP.",
    "2026-04-09: A2A passed 150 supporting organizations, with enterprise production use in its first year.",
    "2026-10-27: a2a-sdk 1.3.0 for Python released; it implements A2A protocol 1.0 with a 0.3 compatibility mode.",
    "2026-10-02: The MCP Python SDK 2.3.0 was released.",
    "2026-09-15: LangGraph 1.2 added improvements to durable execution and checkpointing.",
]


@tool
def search_web(query: str) -> str:
    """Search the web for recent news. Returns matching headlines (newest first)."""
    words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    hits = [n for n in NEWS if words & set(re.findall(r"\w+", n.lower()))]
    return "\n".join(sorted(hits, reverse=True)) or "no results"


research_agent = create_agent(
    llm(),
    tools=[search_web],
    system_prompt=("You are a web research specialist. Always use search_web, then answer in 1-3 sentences "
                   "citing dates. Only state facts found in the search results."),
)


# ----------------------------------------------------------------------------- A2A wrapper
class WebResearchExecutor(AgentExecutor):
    """Bridges A2A tasks to the LangGraph agent and streams progress as task status updates."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is None:  # first message of a new task: publish the Task itself
            await event_queue.enqueue_event(new_task(context.task_id, context.context_id, TaskState.TASK_STATE_SUBMITTED))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        def status(text: str):
            return new_text_message(text, context_id=context.context_id, task_id=context.task_id, role=Role.ROLE_AGENT)

        await updater.start_work(status("Starting web research..."))
        answer = ""
        async for chunk in research_agent.astream({"messages": [HumanMessage(context.get_user_input())]},
                                                  config={"recursion_limit": 8}, stream_mode="updates"):
            for update in chunk.values():
                for m in update.get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            await updater.update_status(TaskState.TASK_STATE_WORKING,
                                                        status(f"Calling {tc['name']}({tc['args']})"))
                    elif isinstance(m, AIMessage) and m.content:
                        answer = m.content if isinstance(m.content, str) else str(m.content)
        await updater.add_artifact([new_text_part(answer)], name="answer")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def build_card(public_url: str) -> AgentCard:
    return AgentCard(
        name="Web Research Agent",
        description="Finds recent news on AI topics and answers with dated, cited facts.",
        version="0.1.0",
        supported_interfaces=[AgentInterface(url=public_url, protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="web_research", name="Web research",
                           description="Answer questions about recent AI news using web search.",
                           tags=["search", "news"], examples=["What's new with the A2A protocol?"])],
    )


class BearerAuth(BaseHTTPMiddleware):
    """Minimal service-to-service auth: protect the RPC endpoint, keep the agent card public."""

    async def dispatch(self, request, call_next):
        token = os.environ.get("A2A_TOKEN")
        if token and request.url.path != "/.well-known/agent-card.json":
            if request.headers.get("authorization") != f"Bearer {token}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build_app(public_url: str) -> Starlette:
    card = build_card(public_url)
    handler = DefaultRequestHandler(agent_executor=WebResearchExecutor(), task_store=InMemoryTaskStore(), agent_card=card)
    app = Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, rpc_url="/")])
    app.add_middleware(BearerAuth)
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9001"))
    url = os.environ.get("PUBLIC_URL", f"http://127.0.0.1:{port}/")  # on Hugging Face: https://<space>.hf.space/
    uvicorn.run(build_app(url), host="0.0.0.0", port=port)
