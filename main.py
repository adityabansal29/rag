import os
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline


def load_env():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in .env")


def main():
    load_env()

    pipeline = RAGPipeline()

    file_path = "document.pdf"
    parser = "unstructured"

    parent_chunks = pipeline.run(file_path=file_path, parser=parser)
    print(f"\nIngested {len(parent_chunks)} sections.")

    results = pipeline.search("your query here", top_k=5)
    for doc in results:
        print(f"\n[{doc.metadata.get('chunk_type')}] {doc.page_content[:200]}")


if __name__ == "__main__":
    main()
