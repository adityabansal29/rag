import asyncio
import os

from dotenv import load_dotenv

from rag.hybrid.pipeline import HybridRAGPipeline
from rag.fusion.pipeline import FusionRAGPipeline
from rag.vectorstores.base import SearchParams
from rag.evaluators.llm_evaluator import LLMEvaluator


def load_env():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in .env")


async def index(file_path: str, parser: str = "unstructured") -> None:
    pipeline = HybridRAGPipeline()
    parent_chunks = await pipeline.run_async(file_path=file_path, parser=parser)
    print(f"\nIngested {len(parent_chunks)} sections.")


def query(questions: list[str], params: SearchParams = None, use_fusion: bool = False) -> None:
    pipeline  = HybridRAGPipeline()
    searcher  = FusionRAGPipeline(pipeline, n_queries=3) if use_fusion else pipeline
    evaluator = LLMEvaluator(pipeline.enricher.llm)
    params    = params or SearchParams(top_k=3)

    for question in questions:
        print(f"\n{'='*60}")
        print(f"  Question: {question}")
        print(f"{'='*60}\n")

        chunks = searcher.search(question, params=params)

        if not chunks:
            print("  No results above similarity threshold.\n")
            continue

        for i, doc in enumerate(chunks):
            print(f"  [{i+1}] score={doc.metadata.get('rrf_score') or doc.metadata.get('score')}  type={doc.metadata.get('chunk_type')}  page={doc.metadata.get('page')}  source={doc.metadata.get('source')}")
            print(f"  {doc.page_content}")
            print()

        answer = pipeline.generate_answer(question, chunks)
        result = evaluator.evaluate_sync(question, chunks, answer)
        result.print_summary()


if __name__ == "__main__":
    load_env()

    # asyncio.run(index("./dynamo.pdf", parser="unstructured"))
    query([
        "Why does Dynamo resolve conflicts during reads rather than writes?",
        "How do vector clocks help Dynamo handle concurrent updates, and what is their limitation?",
        "How does hinted handoff work, and what problem does it solve?"
    ], params=SearchParams(top_k=4, use_hybrid=True), use_fusion=True)
