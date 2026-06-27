def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """
    Reciprocal Rank Fusion across multiple ranked result lists.

    rank_lists: each element is an ordered list of chunk IDs (best first)
    Returns: dict of chunk_id -> RRF score; higher = better
    """
    scores: dict[str, float] = {}
    for ranked_ids in rank_lists:
        for rank, id_ in enumerate(ranked_ids):
            scores[id_] = scores.get(id_, 0.0) + 1 / (k + rank)
    return scores
