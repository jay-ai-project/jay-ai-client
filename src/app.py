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


@cl.cache
def cached_list_ollama_models() -> list[dict]:
    return list_ollama_models()


async def _initialize():
    models = list_ollama_models()
    await cl.ChatSettings([
        Select(
            id="model",
            label="Model",
            items={model["name"]: model["name"] for model in cached_list_ollama_models()},
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
    await cl.context.emitter.set_commands(commands)


@cl.set_chat_profiles
async def set_chat_profiles():
    return [
        cl.ChatProfile(
            name="Jay AI Chat",
            markdown_description="Basic ReAct Agent",
            starters=starters,
            default=True
        ),
        cl.ChatProfile(
            name="Custom Agent",
            markdown_description="Custom Agent",
        )
    ]


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
        "name": t.name,
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


@cl.step(type="tool") 
async def call_mcp_tool(tool_call):
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None
    for k in mcp_tools:
        for mcp_tool in mcp_tools[k]:
            if mcp_tool["name"] == tool_call["name"]:
                mcp_name = k
                break
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    result = await mcp_session.call_tool(tool_call["name"], tool_call["args"])
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

    if cl.context.session.chat_settings.get("tooling"):
        additional_options["mcp_tools"] = []
        for tools in cl.user_session.get("mcp_tools", {}).values():
            additional_options["mcp_tools"].extend(tools)


    with cl.Step(type="llm", name=cl.context.session.chat_settings.get("model"), default_open=True) as graph_step:
        output = cl.Message(content="")

        stream = async_run_stream_messages(
            thread_id=message.id,
            input={"chat_history": chat_history, "messages": messages},
            additional_options=additional_options
        )

        reasoning_step: cl.Step = None
        content_step: cl.Step = None
        tool_call_steps: dict[str, cl.Step] = {}

        while True:    
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

                    if response_metadata := chunk.get("response_metadata"):
                        _step = cl.Step(type="llm", name="Response Metadata")
                        with _step:
                            _step.output = response_metadata
                            await _step.send()
                    
                    # TODO: exclude mcp tool calls
                    if tool_calls := chunk.get("tool_calls"):
                        for tool_call in tool_calls:
                            tool_call_step = cl.Step(name=f"Tool: {tool_call["name"]}", type="tool", language="json", show_input=True)
                            with tool_call_step:
                                tool_call_step.input = tool_call["args"]
                                await tool_call_step.stream_token("")
                            tool_call_steps[tool_call["id"]] = tool_call_step

                    if tool_call_id := chunk.get("tool_call_id"):
                        with tool_call_steps[tool_call_id]:
                            tool_call_steps[tool_call_id].is_error = (chunk["status"] != "success")
                            tool_call_steps[tool_call_id].output = chunk["content"]
                            await tool_call_steps[tool_call_id].send()
                else:
                    break

            if sse.event == "interrupts":
                # TODO: Make this parallel
                tool_results = [await call_mcp_tool(tool_call) for tool_call in sse.json()]
                stream = async_run_stream_messages(
                    thread_id=message.id,
                    input={},
                    resume=tool_results,
                    additional_options=additional_options
                )

            elif sse.event == "error":
                # TODO: handle error
                raise sse.json()
            
            else:
                break
    
    await output.send()
