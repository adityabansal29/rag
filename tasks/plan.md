# Implementation Plan: Fix RAG Pipeline Issues

## Overview

Fix 19 issues from `code-review.md` across the RAG backend. Bugs range from a silent infinite retry loop in the SQS worker to parent summaries being computed but discarded, to brittle free-text parsing in the chat API. Fixes are ordered by blast radius: smallest/most isolated first, shared infrastructure second, data-path changes third, API changes last.

## Architecture Decisions

- **Shared config module** (`backend/config.py`): `state.py` and `sqs_listener.py` have identical `_build_vectorstore()` and client init. Extract once, import everywhere. Do this in Phase 2 so Phase 1 fixes are self-contained.
- **Parent summary as child metadata**: Parent `retrieved_content` stored in each child's LangChain doc metadata at index time (`parent_summary` key). Zero extra storage, no schema change, available at query time. Re-indexing required for existing data.
- **Structured `rag_search` output**: Return a JSON string (list of dicts) instead of free-text. `chat` endpoint parses `json.loads()`. Eliminates the fragile `key=value` header format.
- **Python `logging` module**: Replace all `print()` calls with `logging.getLogger(__name__)`. Log level via `LOG_LEVEL` env var. No third-party dependency.
- **Issue #15 (Chroma BM25 full-corpus scan) deferred**: Fixing this requires an external BM25 index (Elasticsearch, Tantivy, etc.) or storing BM25 indices persistently. Scope is too large for this patch series. Documented as a known limitation.

## Dependency Graph

```
Phase 1 (isolated bug fixes)
    │
    ├── Task 1: Vector store threshold bugs (#3, #4)    [chroma_store.py, pinecone_store.py]
    ├── Task 2: Semaphore private attr (#8)             [enricher.py]
    └── Task 3: SQS retry + shutdown (#1, #19)         [sqs_listener.py]

Phase 2 (shared infra — unblocks Phase 3 worker fix)
    │
    ├── Task 4: Shared config + env vars (#9, #18)     [backend/config.py, state.py, sqs_listener.py]
    ├── Task 5: Async worker (#2)                       [sqs_listener.py] ← needs Task 4
    └── Task 6: Configurable LLM model (#11)           [app.py, pipeline.py, .env.example]

Phase 3 (data correctness)
    │
    ├── Task 7: Chunk ID collision fix (#5)             [models.py]
    └── Task 8: Parent summary persistence (#7)        [pipeline.py, enricher.py]

Phase 4 (API and search logic)
    │
    ├── Task 9:  Structured rag_search output (#10)    [routes.py]
    ├── Task 10: QueryRewriter failure signaling (#6)  [query_rewriter.py, fusion/pipeline.py]
    └── Task 11: SearchParams clarity (#12)            [fusion/pipeline.py]

Phase 5 (observability + performance)
    │
    ├── Task 12: Replace print with logging (#14)      [all files]
    ├── Task 13: DynamoDB Scan pagination (#17)        [routes.py]
    ├── Task 14: Batch enrichment (#16)                [enricher.py]
    └── Task 15: Delete sync wrappers (#13)            [hybrid/pipeline.py, fusion/pipeline.py]
```

---

## Phase 1: Isolated Correctness Bugs

### Task 1: Fix vector store score threshold bugs (Issues #3, #4)

**Description:** Two threshold mismatches: (a) `ChromaVectorStore.bm25_search` ignores `rrf_score_threshold` — it just returns whatever BM25 returns, unlike `_hybrid_search` which filters. (b) `PineconeVectorStore.search` applies `cosine_threshold` to dotproduct scores when `enable_hybrid=True`, which is semantically wrong (dotproduct is unnormalized).

**Acceptance criteria:**
- [ ] `bm25_search` in `chroma_store.py` filters results by `params.rrf_score_threshold` when set
- [ ] `PineconeVectorStore.search` skips score filtering when `enable_hybrid=True` (or uses a separate `hybrid_score_threshold` param — see note)
- [ ] Both methods still return unfiltered results when threshold params are `None`

**Note:** The cleanest fix for Pinecone is to skip `cosine_threshold` filtering entirely when `is_hybrid` (since dotproduct scores have no normalized ceiling). A more correct solution would add a separate `hybrid_score_threshold` to `SearchParams` — use judgment on scope.

**Verification:**
- [ ] Manual: Run Chroma BM25 search with `rrf_score_threshold=0.01` — results below threshold excluded
- [ ] Manual: Run Pinecone hybrid search — no filtering crash or nonsense threshold cutoff
- [ ] `python -c "from rag.vectorstores.chroma_store import ChromaVectorStore"` (import clean)

**Dependencies:** None

**Files:**
- `rag/vectorstores/chroma_store.py` (lines 117-137)
- `rag/vectorstores/pinecone_store.py` (lines 107-121)
- `rag/vectorstores/base.py` (add `hybrid_score_threshold` if going that route)

**Estimated scope:** S (2 files, ~10 lines each)

---

### Task 2: Fix semaphore private attribute access (Issue #8)

**Description:** `LLMEnricher.semaphore` property accesses `self._semaphore._loop` — a private asyncio internal — to detect event loop changes. This breaks on Python 3.10+ where `_loop` was removed from `Semaphore`, and can raise `RuntimeError` on loop transitions. The fix: create the semaphore lazily inside the first `async` call instead of at property access time, or use `asyncio.get_running_loop()` for comparison.

**Acceptance criteria:**
- [ ] No access to `asyncio.Semaphore._loop` (or any `_`-prefixed asyncio internal)
- [ ] Semaphore is properly scoped to the current event loop
- [ ] `max_concurrency` limit still applies

**Verification:**
- [ ] `python -c "from rag.enrichers.enricher import LLMEnricher; e = LLMEnricher(); import asyncio; asyncio.run(e.enrich_all([]))"` runs without RuntimeError

**Dependencies:** None

**Files:**
- `rag/enrichers/enricher.py` (lines 68-72)

**Estimated scope:** XS (1 file, ~5 lines)

---

### Task 3: Fix SQS infinite retry + add graceful shutdown (Issues #1, #19)

**Description:** Two bugs in the same file. (a) On `_process` failure, the `except` block logs but never deletes the SQS message — leading to re-delivery and retry loops. Strategy: delete poison messages after N failures tracked in DynamoDB's `error_count` field, or immediately on first failure (simpler). (b) `listen()` has no signal handler — `Ctrl+C` mid-`asyncio.run` leaves temp files and doesn't drain in-flight work.

**Acceptance criteria:**
- [ ] A permanently failing message (e.g., corrupt file) is NOT re-delivered indefinitely
- [ ] `SIGINT` / `SIGTERM` causes a clean exit after the current message finishes
- [ ] Temp file cleanup still happens on interrupt (the `finally` in `_process` must execute)

**Design choice:** For simplicity, delete the message immediately on any exception (moves retries to application logic / DLQ config). If retry-before-DLQ semantics are needed, add a `max_retries` counter in `error_count` DynamoDB attribute.

**Verification:**
- [ ] Manually trigger a processing failure — message does not reappear after visibility timeout
- [ ] `kill -INT <worker_pid>` — worker exits cleanly, no orphaned temp files

**Dependencies:** None (Task 4 will later refactor this file, but the logic fix is independent)

**Files:**
- `backend/worker/sqs_listener.py` (lines 55-79)

**Estimated scope:** S (1 file, ~20 lines)

---

### Checkpoint: Phase 1

- [ ] All three tasks complete
- [ ] No new imports or dependencies introduced
- [ ] `python -m py_compile` clean on all modified files
- [ ] Manual smoke test: ingest one document, run one query

---

## Phase 2: Shared Infrastructure

### Task 4: Extract shared config + fix env var handling (Issues #9, #18)

**Description:** `state.py` and `sqs_listener.py` have identical `_build_vectorstore()` and AWS client setup. A change to vector store init requires editing both files. Additionally, `os.environ["S3_BUCKET"]` crashes with a `KeyError` at import time — replace with explicit `os.getenv()` + a startup validator that raises a clear `RuntimeError` listing all missing vars.

**Acceptance criteria:**
- [ ] `_build_vectorstore()` defined once in `backend/config.py` (or similar), imported by both `state.py` and `sqs_listener.py`
- [ ] Missing `S3_BUCKET` or `SQS_QUEUE_URL` raises a `RuntimeError` with a human-readable message (not a bare `KeyError`)
- [ ] No behavior change for valid config

**Verification:**
- [ ] Delete `S3_BUCKET` from env, start worker → clear error message, not a traceback
- [ ] `grep -r "_build_vectorstore" backend/` → appears in one file only

**Dependencies:** None (but Phase 3 Task 5 depends on this)

**Files:**
- `backend/config.py` (new)
- `backend/api/state.py`
- `backend/worker/sqs_listener.py`

**Estimated scope:** S (3 files, mostly moves)

---

### Task 5: Make worker async — eliminate per-message event loop (Issue #2)

**Description:** `asyncio.run(pipeline.run_async(...))` inside a `while True` loop creates and destroys an event loop per message. This breaks async resources (connection pools, lazy-initialized clients). Fix: make `listen()` async and `await` the pipeline directly. Entry point becomes `asyncio.run(listen())`.

**Acceptance criteria:**
- [ ] `listen()` is `async def`
- [ ] `_process()` is `async def` and uses `await pipeline.run_async(...)`
- [ ] Entry point: `asyncio.run(listen())`
- [ ] Single event loop lives for the worker's lifetime

**Verification:**
- [ ] Worker starts and processes a message end-to-end without error
- [ ] No `asyncio.run()` inside `_process`

**Dependencies:** Task 4 (shared config module must be in place so refactor is clean)

**Files:**
- `backend/worker/sqs_listener.py`

**Estimated scope:** S (1 file, ~30 lines touched)

---

### Task 6: Make LLM model configurable (Issue #11)

**Description:** `ChatOpenAI(model="gpt-4o")` in `app.py` and `llm_model="gpt-4o"` in `pipeline.py` are both hardcoded. Switching models requires code changes. Fix: read from `LLM_MODEL` env var (defaulting to `"gpt-4o"`). Also: `app.py` instantiates `ChatOpenAI` directly, bypassing the `build_llm_client` abstraction — at minimum, document this gap.

**Acceptance criteria:**
- [ ] `LLM_MODEL` env var controls the model in both `app.py` and `pipeline.py`
- [ ] Default is still `"gpt-4o"` when env var is unset
- [ ] `.env.example` documents `LLM_MODEL`

**Verification:**
- [ ] Set `LLM_MODEL=gpt-4o-mini`, start app, confirm agent uses that model

**Dependencies:** None

**Files:**
- `backend/api/app.py` (line 52)
- `rag/hybrid/pipeline.py` (line 58)
- `.env.example`

**Estimated scope:** XS (2 files, ~3 lines each)

---

### Checkpoint: Phase 2

- [ ] Worker processes messages via single async event loop
- [ ] Missing env vars give clear error messages
- [ ] `_build_vectorstore` defined in one place
- [ ] Full ingestion + query smoke test passes

---

## Phase 3: Data Correctness

### Task 7: Fix chunk ID collision (Issue #5)

**Description:** `Chunk.make_id(source, page, index)` hashes `"{source}:{page}:{index}"` with MD5. The `index` is the parser element index, so re-parsing the same file with different settings can produce the same `(source, page, index)` tuple for different content — silently overwriting the old chunk in the vector store. Fix: include content hash in the ID so chunk identity is tied to content, not just position.

**Acceptance criteria:**
- [ ] `make_id` incorporates a hash of the chunk content (e.g., first 512 chars of `raw_content`)
- [ ] Same content at the same position → same ID (idempotent upsert)
- [ ] Different content at the same position → different ID (no silent overwrite)

**Warning:** This is a **breaking change for existing vector store data** — old chunk IDs will not match new ones, so re-ingesting a file creates duplicate entries rather than updating. Document this in the PR and advise clearing the vector store before re-indexing.

**Verification:**
- [ ] Ingest same file twice → chunk count stays constant (upsert, not duplicate insert)
- [ ] Ingest file, modify content, re-ingest → different chunk IDs (new entries)

**Dependencies:** None (but re-indexing required after deploy)

**Files:**
- `rag/models.py` (lines 46-48)

**Estimated scope:** XS (1 file, ~3 lines)

---

### Task 8: Persist parent summaries in child metadata (Issue #7)

**Description:** `_summarize_parent()` computes a rich LLM summary of each section, but it's never stored anywhere. `chunks_to_langchain_docs` only converts children to LangChain docs — parents are discarded after ingestion. Fix: write `parent.retrieved_content` into each child's metadata as `parent_summary` during `chunks_to_langchain_docs`. Update `generate_answer_async` to include `parent_summary` in the context block.

**Acceptance criteria:**
- [ ] Each stored child document has `metadata["parent_summary"]` set (or `""` if parent has none)
- [ ] `generate_answer_async` includes parent summaries in the context sent to the LLM
- [ ] Children whose parent has no summary (no children, enrichment skipped) get `parent_summary: ""`

**Verification:**
- [ ] After ingestion, query Chroma: `collection.get(include=["metadatas"])` → `parent_summary` key present on documents
- [ ] Query the system — LLM context includes parent section summary alongside child content

**Dependencies:** None (Task 7 runs first but they don't touch the same lines)

**Files:**
- `rag/hybrid/pipeline.py` (lines 22-42 — `chunks_to_langchain_docs`)
- `rag/hybrid/pipeline.py` (lines 206-217 — `generate_answer_async`)

**Estimated scope:** S (1 file, ~15 lines)

---

### Checkpoint: Phase 3

- [ ] Chunk IDs are content-addressed (idempotent re-ingest)
- [ ] Parent summaries appear in stored metadata
- [ ] `generate_answer_async` uses parent context
- [ ] End-to-end: ingest doc, ask question, answer includes section-level context

---

## Phase 4: API and Search Logic

### Task 9: Structured `rag_search` output (Issue #10)

**Description:** `rag_search` tool returns a free-text blob. `chat` endpoint parses it by splitting on `\n\n` and parsing `key=value` headers — fragile and silently swallows errors. Fix: `rag_search` returns `json.dumps(list[dict])`. `chat` endpoint does `json.loads()`. The LLM gets the same text; only the source-extraction in Python changes.

**Acceptance criteria:**
- [ ] `rag_search` returns a JSON string (list of dicts with `score`, `type`, `page`, `source`, `content`)
- [ ] `chat` endpoint extracts sources via `json.loads()`, no string splitting
- [ ] No `except Exception: pass` on source parsing
- [ ] LLM still receives readable context (format the JSON as readable text for the LLM, parse structured for the API response)

**Note:** The LLM sees the tool output as plain text, so format it as readable text for the model but also as parseable JSON for the endpoint. One approach: return JSON where `content` is the readable text block.

**Verification:**
- [ ] POST `/chat` with a question → `sources` array populated correctly
- [ ] Artificially break the tool output format → clear error, not silent empty sources

**Dependencies:** None

**Files:**
- `backend/api/routes.py` (lines 34-47, 135-152)

**Estimated scope:** S (1 file, ~30 lines)

---

### Task 10: QueryRewriter failure signaling (Issue #6)

**Description:** When `rewrite()` fails, it returns `(query, [])` and prints a warning — the caller has no programmatic way to know fusion was degraded. Fix: return a third value or use a typed result; at minimum, the fusion pipeline should set a flag or log at `WARNING` level with enough context for an alert.

**Acceptance criteria:**
- [ ] `FusionRAGPipeline.search_async` returns a result that indicates whether fusion was degraded (e.g., via metadata in the returned docs or a logged structured warning)
- [ ] No silent degradation — downstream consumers can detect the condition

**Design choice (keep minimal):** Add `"fusion_degraded": True` to each returned document's metadata when rewrite fails. This propagates to the API response and can be surfaced to the client without changing the return type signature.

**Verification:**
- [ ] Simulate LLM failure in `rewrite` → returned docs have `fusion_degraded` metadata

**Dependencies:** None

**Files:**
- `rag/fusion/query_rewriter.py`
- `rag/fusion/pipeline.py` (lines 41-46)

**Estimated scope:** S (2 files, ~10 lines)

---

### Task 11: Clarify fusion SearchParams manipulation (Issue #12)

**Description:** `inflated_top_k`, `cosine_threshold=None`, `rrf_score_threshold=None` spread across multiple `replace()` calls is implicit. A new `SearchParams` field would silently pass through with whatever value `replace` sets. Fix: extract a `_fusion_search_params(params, n_queries)` helper that explicitly documents what fusion overrides and why, with a comment on each override.

**Acceptance criteria:**
- [ ] Fusion param overrides are in a single, named function with inline comments
- [ ] No behavior change

**Verification:**
- [ ] Fusion search still returns correct results before and after

**Dependencies:** None

**Files:**
- `rag/fusion/pipeline.py` (lines 53-72)

**Estimated scope:** XS (1 file, ~20 lines refactored)

---

### Checkpoint: Phase 4

- [ ] Chat endpoint returns structured sources
- [ ] Fusion degradation is observable
- [ ] All Phase 1–4 tests pass

---

## Phase 5: Observability and Performance

### Task 12: Replace `print()` with `logging` (Issue #14)

**Description:** All operational `print()` calls replaced with `logging.getLogger(__name__)`. Log level controlled by `LOG_LEVEL` env var (default `INFO`). Debug-tier output (chunk-by-chunk score tables) goes to `DEBUG`, operational status to `INFO`, warnings/errors to appropriate levels.

**Acceptance criteria:**
- [ ] No `print()` calls in `rag/` or `backend/` (except tests / `__main__` blocks)
- [ ] `LOG_LEVEL=DEBUG` shows score tables; `LOG_LEVEL=INFO` shows only summary lines
- [ ] Each module uses `logging.getLogger(__name__)`

**Verification:**
- [ ] `LOG_LEVEL=WARNING python -m backend.worker.sqs_listener` → no score table output
- [ ] `grep -r "^[[:space:]]*print(" rag/ backend/` → zero results

**Dependencies:** None (but easier after other tasks reduce churn)

**Files:**
- `rag/enrichers/enricher.py`
- `rag/vectorstores/chroma_store.py`
- `rag/vectorstores/pinecone_store.py`
- `rag/hybrid/pipeline.py`
- `rag/fusion/pipeline.py`
- `backend/worker/sqs_listener.py`

**Estimated scope:** M (6 files, mechanical substitution)

---

### Task 13: Fix DynamoDB Scan pagination (Issue #17)

**Description:** `GET /jobs` does a full table scan and then Python-truncates to 50 items. With many jobs this wastes RCUs. Fix: use `Limit=50` in the Scan call so DynamoDB returns at most 50 items per request, rather than scanning the entire table.

**Acceptance criteria:**
- [ ] `dynamo.scan()` call includes `Limit=50`
- [ ] Response still sorted by `created_at` descending (note: Scan + Limit doesn't guarantee sort order — sort the 50 returned items)

**Note:** True fix for arbitrary ordering requires a GSI on `created_at`. `Limit=50` only reduces RCU waste; the returned 50 items may not be the 50 most recent. Document this limitation.

**Verification:**
- [ ] `GET /jobs` returns ≤ 50 items
- [ ] DynamoDB CloudWatch metrics show reduced RCU consumption (or manual inspection)

**Dependencies:** None

**Files:**
- `backend/api/routes.py` (line 97)

**Estimated scope:** XS (1 file, 1 line)

---

### Task 14: Batch enrichment to bound memory pressure (Issue #16)

**Description:** `enrich_all` does `asyncio.gather(*[...for child in all_children])` — with 500+ children, 500 coroutines are scheduled immediately. The semaphore limits concurrency but not task count. Fix: process children in batches of `max_concurrency` using `asyncio.gather` on chunks, so at most `max_concurrency` tasks are live at once.

**Acceptance criteria:**
- [ ] At most `max_concurrency * 2` coroutines are live at any point during enrichment
- [ ] All children are still enriched (no silent drops)
- [ ] Batch size defaults to `max_concurrency`

**Verification:**
- [ ] Ingest a document with 100+ non-text chunks — completes without memory spike
- [ ] `len(enriched) == len(all_children)` after `enrich_all`

**Dependencies:** None

**Files:**
- `rag/enrichers/enricher.py` (line 83)

**Estimated scope:** S (1 file, ~15 lines)

---

### Task 15: Delete sync `asyncio.run()` wrapper methods (Issue #13)

**Description:** `HybridRAGPipeline.run()`, `.search()`, `.generate_answer()` and `FusionRAGPipeline.search()`, `.generate_answer()` are `asyncio.run()` wrappers that exist only for `main.py` playground use. They prevent use in async contexts and add ~15 lines of dead weight. Replace `main.py` calls with `asyncio.run(pipeline.run_async(...))` directly.

**Acceptance criteria:**
- [ ] Sync wrapper methods removed from both pipeline classes
- [ ] `main.py` (or any other caller) uses `asyncio.run()` directly
- [ ] No test or production code relies on the removed methods

**Verification:**
- [ ] `grep -r "pipeline\.run\b\|pipeline\.search\b\|pipeline\.generate_answer\b" . --include="*.py"` → zero results (except direct `asyncio.run(` calls)

**Dependencies:** Task 5 (async worker already removed the sync dep from worker)

**Files:**
- `rag/hybrid/pipeline.py` (lines 145-146, 172-177, 219-220)
- `rag/fusion/pipeline.py` (lines 113-118, 123-124)
- `main.py` (any playground scripts)

**Estimated scope:** S (2-3 files, deletion)

---

### Checkpoint: Phase 5 / Final

- [ ] `grep -r "^[[:space:]]*print(" rag/ backend/` → zero results
- [ ] Full ingestion pipeline runs without error
- [ ] Full chat query returns structured sources
- [ ] Worker handles permanent failures without infinite retry
- [ ] All files pass `python -m py_compile`

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Task 7 (chunk ID change) invalidates existing vector store data | High | Clear vector store and re-index all documents after deploy |
| Task 8 (parent summary) increases stored metadata size | Low | Each child gets one string field; Chroma/Pinecone handle this fine |
| Task 5 (async worker) changes process model | Medium | Test with real SQS + S3 before deploying; asyncio + boto3 sync calls need `asyncio.to_thread` |
| Task 9 (structured rag_search) changes LangChain tool output | Medium | Verify LLM still receives readable context; test chat end-to-end |
| Task 15 (delete sync wrappers) may break notebooks | Low | Search all `.py` and `.ipynb` files before deleting |

## Known Limitations (not fixed)

- **Issue #15 (Chroma BM25 full-corpus scan):** Fundamental to in-process BM25 — requires an external index to fix. Deferred.
- **Issue #17 (DynamoDB sort order):** `Scan + Limit=50` reduces RCU but doesn't guarantee most-recent 50. A GSI on `created_at` is the real fix; deferred.
