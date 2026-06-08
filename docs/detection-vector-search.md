# Detection Vector Search

Rhema uses semantic search to catch Bible quotations and paraphrases that do not include an explicit reference like `John 3:16`.

## Runtime Flow

```text
transcript sentence
  -> ONNX embedder
  -> query embedding
  -> HNSW candidate search
  -> exact cosine reranking
  -> top verse candidates
  -> detection merger
```

## Why HNSW

The previous `HnswVectorIndex` type was named as an HNSW index but performed a flat scan over every stored verse vector. That was simple and exact, but it meant every semantic query compared against the full Bible embedding matrix.

The Phase 3 implementation uses HNSW for approximate nearest-neighbor candidate retrieval. HNSW avoids checking every vector by navigating a graph of nearby vectors. This is a common production pattern for semantic search because it can reduce query latency while preserving high recall.

Rhema uses the Rust `hnsw_rs` crate. Its documentation describes the crate as an implementation of the HNSW algorithm and exposes direct insert/search APIs over vector slices. Rhema uses `DistDot` because the embedding pipeline is expected to provide L2-normalized vectors.

Source:

- <https://docs.rs/hnsw_rs/latest/hnsw_rs/hnsw/struct.Hnsw.html>
- <https://docs.rs/hnsw_rs/latest/hnsw_rs/prelude/index.html>

## Exact Reranking

Approximate search is used only to get candidates. Rhema still keeps the original precomputed embeddings in memory.

After HNSW returns candidates, Rhema recomputes exact dot-product similarity between the query and each candidate vector. Because the stored and query embeddings are expected to be L2-normalized, dot product equals cosine similarity.

This gives the app:

- faster candidate retrieval than a full scan
- stable similarity scores
- a simple safety layer against approximate-ordering mistakes

## Index Parameters

Current defaults:

- `m = 16`
- `ef_construction = 200`
- `ef_search = 80`
- max layer = `16`
- rerank candidate multiplier = `8`

The semantic detector usually asks for a small top-k result set. The index searches a larger candidate set and then reranks it exactly.

## Files

Primary implementation:

- `src-tauri/crates/detection/src/semantic/hnsw_index.rs`

Trait boundary:

- `src-tauri/crates/detection/src/semantic/index.rs`

Startup loading:

- `src-tauri/src/lib.rs`

Crate feature:

- `src-tauri/crates/detection/Cargo.toml`

## Notes For Future Work

- Persisting the HNSW graph separately would reduce startup cost after embeddings are loaded.
- The current binary embedding files remain the source of truth for vectors and verse IDs.
- If recall issues appear, increase `ef_search` or the rerank candidate multiplier before changing model thresholds.
- If memory becomes a concern, evaluate quantized vector storage after the regression corpus exists.
