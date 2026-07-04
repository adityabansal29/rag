import json
import os
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv

load_dotenv(override=True)

TABLE  = os.getenv("DYNAMO_TABLE", "rag-jobs")
REGION = os.getenv("AWS_REGION", "us-east-1")

STEP_LABELS = {
    "parsing":   "Parsing",
    "chunking":  "Chunking",
    "enriching": "Enriching",
    "embedding": "Embedding",
    "storing":   "Storing",
}

_dynamo = boto3.client("dynamodb", region_name=REGION)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobTracker:
    def __init__(self, job_id: str, filename: str, s3_key: str):
        self.job_id   = job_id
        self.filename = filename
        self.s3_key   = s3_key
        self._steps: list[dict] = []

    def start(self) -> None:
        """Transition queued → running. Record was already created by the API."""
        _dynamo.update_item(
            TableName=TABLE,
            Key={"job_id": {"S": self.job_id}},
            UpdateExpression="SET #st = :s, updated_at = :u",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={
                ":s": {"S": "running"},
                ":u": {"S": _now()},
            },
        )

    def on_step(self, name: str, status: str, metadata: dict) -> None:
        self._steps.append({
            "name":         name,
            "label":        STEP_LABELS.get(name, name.capitalize()),
            "status":       status,
            "completed_at": _now(),
            "metadata":     metadata,
        })
        _dynamo.update_item(
            TableName=TABLE,
            Key={"job_id": {"S": self.job_id}},
            UpdateExpression="SET steps = :s, updated_at = :u",
            ExpressionAttributeValues={
                ":s": {"S": json.dumps(self._steps)},
                ":u": {"S": _now()},
            },
        )

    def finish(self) -> None:
        _dynamo.update_item(
            TableName=TABLE,
            Key={"job_id": {"S": self.job_id}},
            UpdateExpression="SET #st = :s, updated_at = :u",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={
                ":s": {"S": "done"},
                ":u": {"S": _now()},
            },
        )

    def fail(self, error: str) -> None:
        _dynamo.update_item(
            TableName=TABLE,
            Key={"job_id": {"S": self.job_id}},
            UpdateExpression="SET #st = :s, updated_at = :u, #err = :e",
            ExpressionAttributeNames={"#st": "status", "#err": "error"},
            ExpressionAttributeValues={
                ":s": {"S": "failed"},
                ":u": {"S": _now()},
                ":e": {"S": error},
            },
        )
