# Code Review: RAG Pipeline

## Critical Issues

### 1. Worker silently swallows failed SQS messages
**File:** `backend/worker/sqs_listener.py:61-78`

When `_process` raises, the handler logs the error but does NOT call `msg.receipt` delete. By default SQS auto-re-delivers unacked messages after the visibility timeout, so they'll be picked up again — and fail again — in a loop. If the failure is permanent (corrupt file, bad key), this creates an infinite retry cycle consuming worker resources.

**Fix:** Either call `sqs.delete_message` in the except block (if you want to discard poison messages), or track retry count via DynamoDB `error_count` attribute and move to a DLQ after N attempts.

### 2. `_process()` uses `asyncio.run()` inside a sync `while True` loop
**File:** `backend/worker/sqs_listener.py:52`

`asyncio.run(pipeline.run_async(...))` creates and destroys a new event loop per message. This is legal but wasteful (loop setup/teardown per message). More importantly, if the pipeline ever uses async resources (DB connections, connection pools) those get destroyed every time. It also breaks any module-level lazy-initialized async objects.

**Fix:** Make `listen()` async and use `await pipeline.run_async(...)` directly, or if staying sync, just call `pipeline.run()` (the sync wrapper also calls `asyncio.run` internally, so same overhead).

### 3. `bm25_search` doesn't apply `rrf_score_threshold` filtering
**File:** `rag/vectorstores/chroma_store.py:117-137`

The standalone `bm25_search` method ignores `params.rrf_score_threshold` — it just returns whatever BM25 returns. The `_hybrid_search` method DOES filter by RRF threshold. This is inconsistent: callers using standalone BM25 via `FusionRAGPipeline` get unfiltered results.

### 4. `PineconeVectorStore.search` uses `cosine_threshold` for hybrid/dotproduct scores too
**File:** `rag/vectorstores/pinecone_store.py:107-121`

When `enable_hybrid=True`, the metric is `dotproduct`, but `cosine_threshold` is still used as a score filter. Dotproduct scores are unnormalized (can range from -∞ to ∞), so a 0.6 threshold is semantically meaningless. This will incorrectly filter valid results or pass garbage.

### 5. `Chunk.make_id` uses MD5 — new chunks on same (source, page) collisions
**File:** `rag/models.py:46-48`

`hashlib.md5(f"{source}:{page}:{index}")` — but the `index` parameter is the *element index from the parser*, not a global counter. When you re-parse the same file with a different parser or different strategy settings, the same `(source, page, 0)` tuple can produce a different chunk content with the **same chunk_id**. This means:
- In Chroma: upsert with same `chunk_id` overwrites the old chunk silently
- In Pinecone: same behavior
- No dedup warning or detection

### 6. `QueryRewriter.rewrite` returns `(query, [])` on failure — fusion degenerates silently
**File:** `rag/fusion/query_rewriter.py:37-40`

If the LLM call fails, `rewrite` returns `[]` for variants. `FusionRAGPipeline.search_async` then runs a single-query search with no fusion benefit, printing a warning but with zero structural signaling (no error flag, no metric). The caller (`routes.py:36`) has no way to know fusion was degraded. The search result quality drops silently.

### 7. Parent `retrieved_content` is computed but never persisted — LLM summary work is wasted
**File:** `rag/enrichers/enricher.py:108-128`, `rag/hybrid/pipeline.py:22-42`

Parent chunks get `retrieved_content` via `_summarize_parent()` — an LLM-generated synthesis of all their children. This summary is only in-memory during ingestion. `chunks_to_langchain_docs` only stores child chunks in the vector DB; parents are never persisted anywhere. At query time, there is no way to retrieve the parent summary, so `generate_answer_async` never has access to it. The LLM summarization cost is paid, the output is discarded.

**Fix:** In `chunks_to_langchain_docs`, write `parent.retrieved_content` into each child's metadata (e.g., `"parent_summary": parent.retrieved_content`). The parent summary then rides along with every search result at no extra cost, and `generate_answer_async` can include it as context. Do NOT add parents as separate search-index embeddings — the parent-child split is intentional: children for search precision, parents for answer context.

### 8. `LLMEnricher.semaphore` property checks `_loop` — fragile hack
**File:** `rag/enrichers/enricher.py:68-72`

```python
if self._semaphore is None or self._semaphore._loop != asyncio.get_event_loop():
```

Accessing `asyncio.Semaphore._loop` (a private attribute) to detect a new event loop is fragile Python-internals inspection. If the semaphore was created in a different loop (e.g., async → sync → async cycle in the worker), this check can fail with `RuntimeError: Task <...> got Future attached to a different loop`.

**Fix:** Create the semaphore lazily inside each async method and store nothing module-level. Or just pass `max_concurrency` to `asyncio.gather` with a simpler throttle.

## Code Complexity / Maintainability

### 9. Duplicate `_build_vectorstore()` in `state.py` and `sqs_listener.py`
Both `backend/api/state.py:35-44` and `backend/worker/sqs_listener.py:22-31` have identical `_build_vectorstore()` functions. Also both have duplicated `load_dotenv`, `boto3` client init, and env var reading.

**Fix:** Extract shared config (S3, SQS, DynamoDB client setup, vector store factory) into a single module. Currently a change to the vector store init requires editing two files in lockstep.

### 10. `routes.py` source parsing is brittle and duplicates format knowledge
**File:** `backend/api/routes.py:136-152`

The `/chat` endpoint parses `rag_search` tool output by splitting `msg.content` on `\n\n` and parsing a fragile `key=value` header format. This format is defined by the `rag_search` tool itself (lines 41-46). If the tool's output format changes, source parsing breaks silently (the `except Exception: pass` on line 151 swallows all errors).

**Fix:** Return structured data from `rag_search` (a JSON-serializable object) rather than parsing free-text output.

### 11. `ChatOpenAI` model is hardcoded in two places
**File:** `backend/api/app.py:52` (`model="gpt-4o"`) and `rag/hybrid/pipeline.py:58` (`llm_model="gpt-4o"`)

No env var or config mechanism. Changing the LLM requires code changes in two files. Also inconsistent with `OpenAILLMClient.__init__` default of `"gpt-4o"` — at least that one is consistent, but the app.py instantiation bypasses `build_llm_client` entirely, using `ChatOpenAI` directly.

### 12. `FusionRAGPipeline` search params manipulation is hard to follow
**File:** `rag/fusion/pipeline.py:55-68`

The logic for `inflated_top_k`, `cosine_threshold=None`, `rrf_score_threshold=None` being spread across multiple `replace()` calls is implicit coupling between the fusion layer and what the vector store does with those params. If a new param is added to `SearchParams`, fusion silently passes it through with whatever value `replace` sets — dangerous default behavior.

### 13. Sync `run()` / `search()` methods only exist as `asyncio.run()` wrappers
**File:** `rag/hybrid/pipeline.py:145-146, 172-177`, `rag/fusion/pipeline.py:113-118`

Every async method has a sync twin that just calls `asyncio.run(self.async_version(...))`. These wrappers exist solely for `main.py` playground use. If the playground dropped them, ~15 lines evaporate.

### 14. Print-based logging everywhere
The codebase uses `print()` for operational visibility. There's no log level, no structured logging, no way to silence debug output in production. The vector store search methods are particularly noisy — every query prints tables of chunk IDs and scores to stdout.

## Performance Concerns

### 15. Chroma `_hybrid_search` fetches ALL documents from the collection
**File:** `rag/vectorstores/chroma_store.py:148`

`self._get_bm25_corpus(params.metadata_filters)` calls `self.collection.get(where=...)` which returns every document. For a collection with 100K+ chunks, this loads all embeddings + metadata into memory per query. The `doc_lookup` dict at line 153 holds every chunk in RAM.

### 16. `LLMEnricher.enrich_all` processes ALL children in one `asyncio.gather`
**File:** `rag/enrichers/enricher.py:83`

```python
await asyncio.gather(*[self._enrich_child(child) for child in all_children])
```

With 500+ children and `max_concurrency=10`, 500 coroutines are scheduled simultaneously. While the semaphore limits actual concurrency, the gather still creates 500 tasks with associated overhead. For large documents, this can cause memory pressure.

### 17. No pagination on `GET /jobs` — DynamoDB Scan reads entire table
**File:** `backend/api/routes.py:97`

`state.dynamo.scan(TableName=...)` reads every item in the DynamoDB table, then Python truncates to 50. DynamoDB Scan has a 1MB limit per request, but with many jobs this will require multiple batched scans (Boto3 paginates internally) while the code just takes the first 50 items. It works but wastes RCUs proportional to total job count.

## Shell / Config Issues

### 18. Missing error handling for missing env vars
**File:** `backend/api/state.py:16-17`

`BUCKET = os.environ["S3_BUCKET"]` and `QUEUE_URL = os.environ["SQS_QUEUE_URL"]` use direct key access (will raise `KeyError`) vs `os.getenv()` with defaults like other vars. The app crashes at import time with an unhelpful traceback if these aren't set.

### 19. `sqs_listener.py` has no graceful shutdown
**File:** `backend/worker/sqs_listener.py:64`

The `while True` loop has no signal handler, no way to drain in-flight messages before exit. Ctrl+C during `_process` leaves the temp file on disk (the `finally` won't execute if the SIGINT arrives mid-`asyncio.run`).

---

## Summary

| Severity | Count | Key themes |
|----------|-------|------------|
| Critical | 8 | Infinite retry loop, MD5 collisions, silent degradation, score-type confusion |
| Medium | 6 | Code duplication, hardcoded config, brittle parsing, fragile semaphore |
| Low | 5 | Print logging, sync wrappers, noisy output, DynamoDB Scan, shutdown |