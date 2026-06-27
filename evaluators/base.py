from pydantic import BaseModel


class ChunkScore(BaseModel):
    chunk_id: str
    relevance: float   # 0 = unrelated, 1 = directly answers query
    reason: str


class EvalResult(BaseModel):
    query: str
    answer: str
    chunk_scores: list[ChunkScore]
    avg_chunk_relevance: float    # 0-1
    faithfulness: float           # 0 = hallucinated, 1 = fully grounded
    unsupported_claims: list[str]
    faithfulness_reason: str

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  RAG Evaluation")
        print(f"{'='*60}")
        print(f"  Query     : {self.query}")
        print(f"  Answer    : {self.answer[:120]}{'...' if len(self.answer) > 120 else ''}")

        print(f"\n  Chunk Relevance  (avg: {self.avg_chunk_relevance:.2f})")
        print(f"  {'Chunk ID':<40} {'Score':>6}  Reason")
        print(f"  {'-'*40} {'-'*6}  {'-'*30}")
        for cs in self.chunk_scores:
            print(f"  {cs.chunk_id:<40} {cs.relevance:>6.2f}  {cs.reason}")

        print(f"\n  Faithfulness     : {self.faithfulness:.2f}")
        print(f"  Reason           : {self.faithfulness_reason}")
        if self.unsupported_claims:
            print(f"  Unsupported claims:")
            for claim in self.unsupported_claims:
                print(f"    - {claim}")
        print()
