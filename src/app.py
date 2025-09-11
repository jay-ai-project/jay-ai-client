import textwrap
from typing import Any
import asyncio

import chainlit as cl
from chainlit.mcp import McpConnection
from chainlit.data import get_data_layer
from chainlit.types import ThreadDict
from chainlit.input_widget import Select, Switch, Slider, TextInput
from chainlit.element import Element
from mcp import ClientSession

from chat import async_run_stream_messages, list_ollama_models
import utils


########################################################################################################################
# Authorization
########################################################################################################################
@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
    return None


########################################################################################################################
# Configuration
########################################################################################################################
starters = [
    cl.Starter(
        label="Morning routine ideation",
        message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
        icon="/public/idea.svg",
    ),
    cl.Starter(
        label="Explain superconductors",
        message="Explain superconductors like I'm five years old.",
        icon="/public/learn.svg",
    ),
    cl.Starter(
        label="Python script for daily email reports",
        message="Write a script to automate sending daily email reports in Python, and walk me through how I would set it up.",
        icon="/public/terminal.svg",
        command="code",
    ),
    cl.Starter(
        label="Text inviting friend to wedding",
        message="Write a text asking a friend to be my plus-one at a wedding next month. I want to keep it super short and casual, and offer an out.",
        icon="/public/write.svg",
    )
]

commands = [
    {"id": "Clear Chat History", "icon": "pencil-off", "description": "Delete past chat history"},
    # {"id": "Picture", "icon": "image", "description": "Use ComfyUI"},
    # {"id": "Search", "icon": "globe", "description": "Find on the web"},
    # {"id": "JiraTicket", "icon": "pen-line", "description": "Make a Jira ticket"},
]

agents = {
    "Ollama Agent": {
        "graph_name": "Ollama_Agent",
        "chat_profile": {
            "markdown_description": "Basic ReAct Agent",
            "starters": starters,
            "default": True
        },
    },
    "Test Code Writer": {
        "graph_name": "Test_Code_Writer",
        "chat_profile": {
            "markdown_description": textwrap.dedent("""
                ### 테스트 코드 자동 생성

                소스 코드를 분석하여 단위 테스트 코드를 자동으로 생성하는 AI 에이전트입니다. 테스트 계획을 수립하고, 계획에 따라 코드를 작성한 후, 최종 작업 내용을 보고합니다. 반복적인 테스트 코드 작성 업무를 자동화하여 개발 생산성을 높일 수 있습니다.
            """),
        },
    }
}


@cl.set_chat_profiles
async def set_chat_profiles():
    return [cl.ChatProfile(name, **agent["chat_profile"]) for name, agent in agents.items()]


@cl.cache
def cached_list_ollama_models() -> list[dict]:
    return list_ollama_models()


async def _initialize():
    if cl.context.session.chat_profile == "Ollama Agent":
        models = cached_list_ollama_models()
        await cl.ChatSettings([
            Select(
                id="model",
                label="Model",
                items={model["name"]: model["name"] for model in models},
                initial_value=cl.context.session.chat_settings.get("model", models[0]["name"])
            ),
            Slider(
                id="num_ctx",
                label="Context Window",
                min=4*1024,
                max=128*1024,
                step=4*1024,
                initial=cl.context.session.chat_settings.get("num_ctx", 16*1024)
            ),
            Switch(
                id="reasoning",
                label="Reasoning",
                initial=cl.context.session.chat_settings.get("reasoning", False)
            ),
            Switch(
                id="tooling",
                label="Tool Use",
                initial=cl.context.session.chat_settings.get("tooling", False)
            ),
            TextInput(
                id="system_prompt",
                label="System Prompt",
                initial=cl.context.session.chat_settings.get("system_prompt"),
                multiline=True
            )
        ]).send()
    elif cl.context.session.chat_profile == "Test Code Writer":
        await cl.ChatSettings([
            Slider(
                id="num_ctx",
                label="Context Window",
                min=4*1024,
                max=128*1024,
                step=4*1024,
                initial=cl.context.session.chat_settings.get("num_ctx", 32*1024)
            ),
            Switch(
                id="reasoning",
                label="Reasoning",
                initial=cl.context.session.chat_settings.get("reasoning", False)
            )
        ]).send()
    
    await cl.context.emitter.set_commands(commands)


@cl.on_chat_start
async def on_chat_start():
    await _initialize()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    await _initialize()


########################################################################################################################
# MCP Tools
########################################################################################################################
@cl.on_mcp_connect
async def on_mcp_connect(connection: McpConnection, session: ClientSession):
    # List available tools
    result = await session.list_tools()
    
    # Process tool metadata
    tools = [{
        "name": f"mcp__{connection.name}__{t.name}",
        "description": t.description,
        "input_schema": t.inputSchema,
    } for t in result.tools]
    
    # Store tools for later use
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    mcp_tools = cl.user_session.get("mcp_tools", {})
    if name in mcp_tools:
        del mcp_tools[name]
    cl.user_session.set("mcp_tools", mcp_tools)


async def call_mcp_tool(tool_call: dict[str, str | Any]):
    splits = tool_call["name"].split("__")
    mcp_name = splits[1]
    tool_name = "__".join(splits[2:])
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    result = await mcp_session.call_tool(tool_name, tool_call["args"])
    return [content.model_dump() for content in result.content]


########################################################################################################################
# Handle messages
########################################################################################################################
@cl.cache
def cached_read_base64_from_url(url: str) -> str:
    return utils.read_base64_from_url(url)


async def get_chat_history(limit: int | None = None, exclude_id: str | None = None):
    params = {"thread_id": cl.context.session.thread_id}
    query = """
        SELECT 
            id, type, output, "isError" 
        FROM "Step" 
        WHERE "threadId" = $1
        AND type IN ('user_message', 'assistant_message')
    """
    if exclude_id:
        query += " AND id <> $2"
        params["id"] = exclude_id
    query += ' ORDER BY "startTime" DESC'
    if limit:
        query += f" LIMIT {limit}"
    
    data_layer = get_data_layer()
    steps = reversed(await data_layer.execute_query(query, params))
    elements = await data_layer.execute_query(f"""
        SELECT 
            e."stepId", e.mime, e."objectKey"
        FROM "Element" e JOIN ({query}) s
        ON e."stepId" = s.id
    """, params)

    async def _async_mapper(element):
        url = await data_layer.storage_client.get_read_url(element["objectKey"])
        url = url.split("?")[0] # for local caching
        base64data = cached_read_base64_from_url(url)
        type = Element.infer_type_from_mime(element["mime"])
        return element["stepId"], {"type": type, "mime_type": element["mime"], "source_type": "base64", "data": base64data}

    elements = await asyncio.gather(*list(map(_async_mapper, elements)))
    elements_for_step: dict[str, list] = {}
    for step_id, element in elements:
        if step_id not in elements_for_step:
            elements_for_step[step_id] = []
        elements_for_step[step_id].append(element)
    
    # TODO: Tool Message에 대한 처리, metadata의 tool_calls, tool message 등 처리
    def _mapper(step):
        content = []
        if step["output"]:
            content.append({"type": "text", "text": step["output"]})
        if elements := elements_for_step.get(step["id"]):
            content.extend(elements)
        if step["type"] == "user_message":
            return {"type": "human", "content": content}
        if step["type"] == "assistant_message":
            return {"type": "ai", "content": content}
    
    result = list(map(_mapper, steps))
    return result


async def clear_chat_history():
    data_layer = get_data_layer()
    thread = await data_layer.get_thread(cl.context.session.thread_id)
    for element in thread["elements"]:
        await data_layer.delete_element(element["id"], thread["id"])
    for step in thread["steps"]:
        await data_layer.delete_step(step["id"])


async def get_content_with_elements(message: cl.Message) -> list[dict]:
    content = []
    if message.content:
        content.append({
            "type": "text",
            "text": message.content
        })
    for element in message.elements:
        content.append({
            "type": element.type,
            "mime_type": element.mime,
            "source_type": "base64",
            "data": utils.read_base64_from_file_path(element.path)
        })
    return content


@cl.on_message
async def on_message(message: cl.Message):
    if message.command == "Clear Chat History":
        message.type = "system_message"
        message.content = "chat history deleted."
        await clear_chat_history()
        await message.update()
        return
    
    chat_history = await get_chat_history(exclude_id=message.id, limit=100)

    messages = [{
        "type": "human",
        "content": await get_content_with_elements(message)
    }]

    additional_options={
        "ollama": dict(
            model=cl.context.session.chat_settings.get("model"),
            num_ctx=cl.context.session.chat_settings.get("num_ctx"),
            reasoning=cl.context.session.chat_settings.get("reasoning"),
            tooling=cl.context.session.chat_settings.get("tooling")
        ),
        "system_prompt": cl.context.session.chat_settings.get("system_prompt"),
    }
    
    additional_options["mcp_tools"] = []
    for tools in cl.user_session.get("mcp_tools", {}).values():
        additional_options["mcp_tools"].extend(tools)

    graph_step_name = cl.context.session.chat_settings.get("model") or "Graph"
    with cl.Step(type="llm", name=graph_step_name, default_open=True):
        output = cl.Message(content="")

        stream = async_run_stream_messages(
            graph_name=agents[cl.context.session.chat_profile]["graph_name"],
            thread_id=message.id,
            input={"chat_history": chat_history, "messages": messages},
            additional_options=additional_options
        )

        while True:    
            reasoning_step: cl.Step = None
            content_step: cl.Step = None
            interrupts: list[dict] = []

            async for sse in stream:
                if sse.event == "message":
                    chunk: dict = sse.json()["chunk"]

                    if reasoning_content := chunk["additional_kwargs"].get("reasoning_content"):
                        if not reasoning_step:
                            reasoning_step = cl.Step(name="Reasoning", type="llm")
                        with reasoning_step:
                            await reasoning_step.stream_token(reasoning_content)
                        continue
                    
                    if reasoning_step:
                        await reasoning_step.send()
                        reasoning_step = None
                    
                    if (content := chunk.get("content")) and not chunk.get("tool_call_id"):
                        if not content_step:
                            content_step = cl.Step(name="Content", type="llm")
                        with content_step:
                            await content_step.stream_token(content)
                        continue

                    if content_step:
                        output.content = content_step.output
                        await content_step.send()
                        content_step = None
                
                elif sse.event == "interrupts":
                    interrupts = sse.json()
                elif sse.event == "error":
                    # TODO: handle error
                    raise sse.json()

            if interrupts:
                resume_map = {}
                
                for interrupt in interrupts:
                    interrupt_id = interrupt["id"]
                    interrupt_value = interrupt["value"]

                    if interrupt_value["type"] == "text":
                        res = await cl.AskUserMessage(content=interrupt_value["text"]).send()
                        resume_map[interrupt_id] = res["output"]
                        
                    elif interrupt_value["type"] == "tool_calls":
                        tool_results = []
                        for tool_call in interrupt_value["tool_calls"]:
                            tool_result = None

                            with cl.Step(name=f"Tool: {tool_call["name"]}", type="tool") as step:
                                step.input = tool_call["args"]
                                if "create" in tool_call["name"] or "write" in tool_call["name"]:
                                    step.default_open = True
                                    await step.update()
                                    
                                    res = await cl.AskActionMessage(
                                        content="Tool call", 
                                        actions=[
                                            cl.Action(name="Accept", payload={"value": "accept"}, label="✅ Accept"),
                                            cl.Action(name="Cancel", payload={"value": "cancel"}, label="❌ Cancel")
                                        ]
                                    ).send()
                                    if res.get("payload").get("value") == "cancel":
                                        step.is_error = True
                                        tool_result = "Tool calling was canceled by the user."

                            if not tool_result:
                                tool_result = await call_mcp_tool(tool_call)

                            step.output = tool_result
                            tool_results.append(tool_result)
                            await step.update()

                        resume_map[interrupt_id] = tool_results

                stream = async_run_stream_messages(
                    graph_name=agents[cl.context.session.chat_profile]["graph_name"],
                    thread_id=message.id,
                    input={},
                    additional_options=additional_options,
                    resume=resume_map
                )
            else:
                break
    
    await output.send()
