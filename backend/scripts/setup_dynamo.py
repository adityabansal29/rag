"""Creates all required DynamoDB tables. Safe to run multiple times."""
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(override=True)

REGION = os.getenv("AWS_REGION", "us-east-1")
dynamo = boto3.client("dynamodb", region_name=REGION)


def create_table(name: str, key_schema: list, attr_defs: list) -> None:
    try:
        resp = dynamo.create_table(
            TableName=name,
            KeySchema=key_schema,
            AttributeDefinitions=attr_defs,
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"Created: {resp['TableDescription']['TableArn']}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"Table '{name}' already exists — skipping.")
        else:
            raise


# Jobs table
create_table(
    name=os.getenv("DYNAMO_TABLE", "rag-jobs"),
    key_schema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
    attr_defs=[{"AttributeName": "job_id", "AttributeType": "S"}],
)

# LangGraph chat checkpoints table
# pk/sk values are what langgraph_checkpoint_dynamodb hardcodes in its queries
pk, sk = "thread_id", "checkpoint_id"
create_table(
    name=os.getenv("CHAT_CHECKPOINTS_TABLE", "rag-chat-checkpoints"),
    key_schema=[
        {"AttributeName": pk, "KeyType": "HASH"},
        {"AttributeName": sk, "KeyType": "RANGE"},
    ],
    attr_defs=[
        {"AttributeName": pk, "AttributeType": "S"},
        {"AttributeName": sk, "AttributeType": "S"},
    ],
)

# LangGraph chat writes table
pk, sk = "thread_id_checkpoint_id_checkpoint_ns", "task_id_idx"
create_table(
    name=os.getenv("CHAT_WRITES_TABLE", "rag-chat-writes"),
    key_schema=[
        {"AttributeName": pk, "KeyType": "HASH"},
        {"AttributeName": sk, "KeyType": "RANGE"},
    ],
    attr_defs=[
        {"AttributeName": pk, "AttributeType": "S"},
        {"AttributeName": sk, "AttributeType": "S"},
    ],
)
