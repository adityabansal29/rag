import asyncio
import os

from dotenv import load_dotenv

from rag.pipeline import RAGPipeline
from rag.vectorstores.base import SearchParams


def load_env():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in .env")


async def index(file_path: str, parser: str = "unstructured") -> None:
    pipeline = RAGPipeline()
    parent_chunks = await pipeline.run_async(file_path=file_path, parser=parser)
    print(f"\nIngested {len(parent_chunks)} sections.")


def query(questions: list[str], params: SearchParams = None) -> None:
    pipeline = RAGPipeline()
    params = params or SearchParams(top_k=3)

    for question in questions:
        print(f"\n{'='*60}")
        print(f"  Question: {question}")
        print(f"{'='*60}\n")

        results = pipeline.search(question, params=params)

        if not results:
            print("  No results above similarity threshold.\n")
            continue

        for i, doc in enumerate(results):
            print(f"  [{i+1}] score={doc.metadata.get('score')}  type={doc.metadata.get('chunk_type')}  page={doc.metadata.get('page')}  source={doc.metadata.get('source')}")
            print(f"  {doc.page_content}")
            print()


if __name__ == "__main__":
    load_env()

    # asyncio.run(index("./dynamo.pdf", parser="unstructured"))
    query([
        "Why does Dynamo resolve conflicts during reads rather than writes?",
        "How do vector clocks help Dynamo handle concurrent updates, and what is their limitation?",
        "How does hinted handoff work, and what problem does it solve?"
    ], params=SearchParams(top_k=4, use_hybrid=True))
