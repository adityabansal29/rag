import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from rag.vectorstores.base import SearchParams
from backend.api import state

router = APIRouter()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_job(item: dict) -> dict:
    return {
        "job_id":     item["job_id"]["S"],
        "filename":   item.get("filename", {}).get("S", ""),
        "s3_key":     item.get("s3_key", {}).get("S", ""),
        "status":     item.get("status", {}).get("S", "unknown"),
        "steps":      json.loads(item.get("steps", {}).get("S", "[]")),
        "created_at": item.get("created_at", {}).get("S", ""),
        "updated_at": item.get("updated_at", {}).get("S", ""),
        "error":      item.get("error", {}).get("S") or None,
    }


@tool
async def rag_search(query: str) -> str:
    """Search the knowledge base using fusion RAG and return relevant chunks with scores."""
    _, chunks = await state.searcher.search_async(query, params=SearchParams(top_k=4, use_hybrid=True))
    if not chunks:
        return "No relevant information found."
    parts = []
    for i, doc in enumerate(chunks):
        score = doc.metadata.get("rrf_score") or doc.metadata.get("score", 0)
        parts.append(
            f"[{i+1}] score={score} type={doc.metadata.get('chunk_type', 'text')} "
            f"page={doc.metadata.get('page', '')} source={doc.metadata.get('source', '')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.get("/presigned-url")
def get_presigned_url(filename: str):
    key = f"uploads/{uuid.uuid4()}/{filename}"
    url = state.s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": state.BUCKET, "Key": key},
        ExpiresIn=state.PRESIGNED_EXPIRY,
    )
    return {"url": url, "key": key}


class NotifyRequest(BaseModel):
    key: str
    filename: str = ""


@router.post("/notify")
def notify(req: NotifyRequest):
    job_id   = str(uuid.uuid4())
    filename = req.filename or req.key.split("/")[-1]
    now      = _now()

    state.dynamo.put_item(
        TableName=state.DYNAMO_TABLE,
        Item={
            "job_id":     {"S": job_id},
            "filename":   {"S": filename},
            "s3_key":     {"S": req.key},
            "status":     {"S": "queued"},
            "steps":      {"S": "[]"},
            "created_at": {"S": now},
            "updated_at": {"S": now},
        },
    )
    state.sqs.send_message(
        QueueUrl=state.QUEUE_URL,
        MessageBody=json.dumps({"s3_key": req.key, "job_id": job_id, "filename": filename}),
    )
    return {"job_id": job_id, "status": "queued"}


# ── Jobs ───────────────────────────────────────────────────────────────────────

@router.get("/jobs")
def list_jobs():
    resp = state.dynamo.scan(TableName=state.DYNAMO_TABLE)
    jobs = [_parse_job(item) for item in resp.get("Items", [])]
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return jobs[:50]


@router.get("/job/{job_id}")
def get_job(job_id: str):
    resp = state.dynamo.get_item(
        TableName=state.DYNAMO_TABLE,
        Key={"job_id": {"S": job_id}},
    )
    item = resp.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="Job not found")
    return _parse_job(item)


# ── Chat ───────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str


@router.post("/chat")
async def chat(req: ChatRequest):
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    config = {"configurable": {"thread_id": req.thread_id}}
    result = await state.agent.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )

    answer = result["messages"][-1].content

    sources = []
    for msg in result["messages"]:
        if getattr(msg, "name", None) == "rag_search":
            for block in msg.content.split("\n\n"):
                if not block.startswith("["):
                    continue
                try:
                    header, *body_lines = block.split("\n")
                    kv = dict(p.split("=", 1) for p in header.split() if "=" in p)
                    sources.append({
                        "content": "\n".join(body_lines).strip(),
                        "score":   float(kv.get("score", 0)),
                        "type":    kv.get("type", ""),
                        "page":    kv.get("page", ""),
                        "source":  kv.get("source", ""),
                    })
                except Exception:
                    pass

    return {"answer": answer, "sources": sources}


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok"}
