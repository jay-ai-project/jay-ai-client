from typing import Any, AsyncIterator

import httpx
from httpx_sse import aconnect_sse, ServerSentEvent

import config


async def async_run_stream_messages(
    graph_name: str,
    thread_id: str, 
    input: dict,
    resume: Any | None = None,
    additional_options: dict = {}
) -> AsyncIterator[ServerSentEvent]:
    url = f"http://localhost:8123/api/chat/stream"
    payload = {
        "graph_name": graph_name,
        "input": input,
        "resume": resume,
        "config": {
            "configurable": {
                "thread_id": thread_id,
                "additional_options": additional_options
            }
        }
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with aconnect_sse(client, "POST", url, json=payload) as event_source:
            event_source.response.raise_for_status()
            async for sse in event_source.aiter_sse():
                yield sse


def list_ollama_models() -> list[dict]:
    url = f"{config.OLLAMA_BASE_URL}/api/tags"
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()["models"]
