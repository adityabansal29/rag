import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph_checkpoint_dynamodb.saver import DynamoDBSaver

from backend.api import state
from backend.api.routes import router, rag_search


class AsyncDynamoDBSaver(DynamoDBSaver):
    """Wraps sync DynamoDBSaver methods with asyncio.to_thread for async compatibility."""

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await asyncio.to_thread(self.put_writes, config, writes, task_id)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        loop = asyncio.get_event_loop()
        items = await loop.run_in_executor(None, lambda: list(self.list(config, filter=filter, before=before, limit=limit)))
        for item in items:
            yield item


def _make_dynamo_checkpointer() -> AsyncDynamoDBSaver:
    return AsyncDynamoDBSaver(
        client_config={"region_name": state.AWS_REGION},
        checkpoints_table_name=state.CHAT_CHECKPOINTS_TABLE,
        writes_table_name=state.CHAT_WRITES_TABLE,
    )


_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant with access to a document knowledge base. "
    "Always use the rag_search tool to answer questions — do not rely on prior knowledge alone. "
    "Base your answers strictly on the retrieved context. "
    "If the search returns no relevant results, say so clearly rather than guessing."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o"), temperature=0)
    if state.CHECKPOINTER_BACKEND == "dynamodb":
        saver = _make_dynamo_checkpointer()
        state.agent = create_agent(llm, tools=[rag_search], checkpointer=saver, system_prompt=_SYSTEM_PROMPT)
        yield
    else:
        async with AsyncSqliteSaver.from_conn_string(state.CHECKPOINTER_DB) as saver:
            state.agent = create_agent(llm, tools=[rag_search], checkpointer=saver, system_prompt=_SYSTEM_PROMPT)
            yield


app = FastAPI(title="RAG Ingestion API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
