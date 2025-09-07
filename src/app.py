import chainlit as cl
from chainlit.config import (
    ChainlitConfigOverrides,
    SpontaneousFileUploadFeature,
    McpFeature,
    UISettings,
    FeaturesSettings
)
from chainlit.mcp import McpConnection
from mcp import ClientSession

from chainlit.data import get_data_layer
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from chainlit.types import ThreadDict
from chainlit.user_session import UserSession
from chainlit.session import WebsocketSession
from chainlit.context import ChainlitContext
from chainlit.input_widget import Select, Switch, Slider, InputWidget
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.schema.runnable.config import RunnableConfig
from contextlib import nullcontext
from langchain_core.runnables import RunnableSerializable
from langchain_core.messages import AIMessageChunk
import httpx

import config


########################################################################################################################
# Authorization
########################################################################################################################
@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
    if (username, password) == ("jay", "jay"):
        return cl.User(identifier="jay", metadata={"role": "user", "provider": "credentials"})
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
    {"id": "Picture", "icon": "image", "description": "Use DALL-E"},
    {"id": "Search", "icon": "globe", "description": "Find on the web"},
    {"id": "JiraTicket", "icon": "pen-line", "description": "Make a Jira ticket"},
]


@cl.cache
def list_ollama_models() -> list[dict]:
    url = f"{config.OLLAMA_BASE_URL}/api/tags"
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()["models"]


async def _initialize():
    models = list_ollama_models()
    await cl.ChatSettings([
        Select(
            id="model",
            label="Model",
            items={model["name"]: model["name"] for model in list_ollama_models()},
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
async def get_chat_history(limit: int | None = None, exclude_id: str | None = None):
    params = {"thread_id": cl.context.session.thread_id}
    query = """
        SELECT 
            type, output, "isError" 
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
    result = reversed(await data_layer.execute_query(query, params))
    
    # TODO: Tool Message에 대한 처리, metadata의 tool_calls, tool message 등 처리
    # TODO: elements 데이터 처리, cache 활용
    def _mapper(message):
        if message["type"] == "user_message":
            return {"type": "human", "content": message["output"]}
        if message["type"] == "assistant_message":
            return {"type": "ai", "content": message["output"]}
    
    return list(map(_mapper, result))


@cl.on_message
async def on_message(message: cl.Message):
    # TODO: message.command 로 특정 작업 지시 추가
    
    # Resume 후 첫 메시지 입력 시 history에 현재 메시지도 포함되어 exclude 처리 필요함
    # TODO: Context 크기에 따른 limit 조정, 매뉴얼 설정, 이전 대화 기록 삭제 등 처리 등
    # TODO: input element 데이터 처리
    chat_history = await get_chat_history(exclude_id=message.id, limit=100)
    print(message.elements)
    chat_history.append({
        "type": "human",
        "content": message.content
    })

    llm = ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=cl.context.session.chat_settings.get("model"),
        num_ctx=cl.context.session.chat_settings.get("num_ctx"),
        reasoning=cl.context.session.chat_settings.get("reasoning"),
    )

    if cl.context.session.chat_settings.get("tooling"):
        mcp_tools: dict = cl.user_session.get("mcp_tools", {})
        for tools in mcp_tools.values():
            llm = llm.bind_tools(tools=tools)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            # SystemMessage(SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    runnable = prompt_template | llm
    
    while True:
        ai_message = await chat_stream(runnable, chat_history)
        chat_history.append(ai_message)
        if not ai_message["tool_calls"]:
            break
        for tool_call in ai_message["tool_calls"]:
            tool_result = await call_mcp_tool(tool_call)
            # TODO: tool_result.isError handling
            chat_history.append({
                "type": "tool",
                "content": tool_result,
                "tool_call_id": tool_call["id"]
            })


async def chat_stream(runnable: RunnableSerializable[dict, AIMessageChunk], messages: list[dict]):
    output = cl.Message(content="")

    if cl.context.session.chat_settings.get("reasoning"):
        reasoning_step = cl.Step("Reasoning", type="llm", default_open=True)
    else:
        reasoning_step = nullcontext()

    tool_calls = []
    response_metadata = {}

    async with reasoning_step:
        async for chunk in runnable.astream({"messages": messages}):
            if reasoning_chunk := chunk.additional_kwargs.get("reasoning_content"):
                await reasoning_step.stream_token(reasoning_chunk)
            if content_chunk := chunk.content:
                await output.stream_token(content_chunk)
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            if chunk.response_metadata:
                response_metadata = chunk.response_metadata

    if isinstance(reasoning_step, cl.Step) and not reasoning_step.output:
        await reasoning_step.remove()
    if content := output.content:
        await output.send()
    else:
        await output.remove()
    
    return {
        "type": "ai",
        "content": content,
        "tool_calls": tool_calls,
        "response_metadata": response_metadata
    }
