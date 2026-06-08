//! HNSW cosine-similarity vector index with exact reranking.
//!
//! HNSW provides a fast approximate candidate set.  We keep the original
//! embeddings in memory and rerank the returned candidates with exact cosine
//! similarity so the public score remains the same type of score the previous
//! flat index returned.
//!
//! This module is only compiled when the `vector-search` feature is enabled.

#[cfg(feature = "vector-search")]
use std::path::Path;

#[cfg(feature = "vector-search")]
use super::index::{SearchResult, VectorIndex};
#[cfg(feature = "vector-search")]
use crate::error::DetectionError;
#[cfg(feature = "vector-search")]
use hnsw_rs::prelude::{DistDot, Hnsw};

#[cfg(feature = "vector-search")]
const HNSW_M: usize = 16;
#[cfg(feature = "vector-search")]
const HNSW_EF_CONSTRUCTION: usize = 200;
#[cfg(feature = "vector-search")]
const HNSW_EF_SEARCH: usize = 80;
#[cfg(feature = "vector-search")]
const RERANK_MULTIPLIER: usize = 8;
#[cfg(feature = "vector-search")]
const HNSW_MAX_LAYER: usize = 16;

/// Vector index backed by HNSW plus original pre-computed embeddings.
///
/// HNSW search returns approximate nearest neighbours.  The final results are
/// reranked by exact dot product.  Because all stored vectors and query vectors
/// are expected to be L2-normalised, the dot product equals cosine similarity.
#[cfg(feature = "vector-search")]
pub struct HnswVectorIndex {
    /// Approximate nearest-neighbour graph keyed by row index.
    hnsw: Hnsw<'static, f32, DistDot>,
    /// Flattened embedding matrix: `embeddings[i * dim .. (i+1) * dim]`
    /// is the vector for verse `verse_ids[i]`.
    embeddings: Vec<f32>,
    /// Verse (or row) identifiers, one per stored vector.
    verse_ids: Vec<i64>,
    /// Dimensionality of each embedding vector.
    dimension: usize,
}

#[cfg(feature = "vector-search")]
impl HnswVectorIndex {
    /// Load pre-computed embeddings and their verse IDs from binary files.
    ///
    /// **Embeddings file** — a sequence of `f32` values in native byte
    /// order.  Each consecutive `dim` floats form one vector.
    ///
    /// **IDs file** — a sequence of `i64` values in native byte order,
    /// one per vector.
    pub fn load(
        embeddings_path: &Path,
        ids_path: &Path,
        dim: usize,
    ) -> Result<Self, DetectionError> {
        // --- Read embeddings ---
        let emb_bytes = std::fs::read(embeddings_path).map_err(|e| {
            DetectionError::Internal(format!(
                "read embeddings {}: {e}",
                embeddings_path.display()
            ))
        })?;

        if emb_bytes.len() % std::mem::size_of::<f32>() != 0 {
            return Err(DetectionError::Internal(
                "embeddings file size is not a multiple of 4".into(),
            ));
        }

        let embeddings: Vec<f32> = bytemuck::cast_slice(&emb_bytes).to_vec();
        let num_vectors = embeddings.len() / dim;

        if embeddings.len() % dim != 0 {
            return Err(DetectionError::Internal(format!(
                "embeddings length {} is not a multiple of dim {}",
                embeddings.len(),
                dim
            )));
        }

        // --- Read IDs ---
        let ids_bytes = std::fs::read(ids_path).map_err(|e| {
            DetectionError::Internal(format!("read ids {}: {e}", ids_path.display()))
        })?;

        if ids_bytes.len() % std::mem::size_of::<i64>() != 0 {
            return Err(DetectionError::Internal(
                "ids file size is not a multiple of 8".into(),
            ));
        }

        let verse_ids: Vec<i64> = bytemuck::cast_slice(&ids_bytes).to_vec();

        if verse_ids.len() != num_vectors {
            return Err(DetectionError::Internal(format!(
                "vector count mismatch: {} embeddings vs {} ids",
                num_vectors,
                verse_ids.len()
            )));
        }

        log::info!(
            "HnswVectorIndex loaded: {} vectors, dim={}",
            num_vectors,
            dim
        );

        Self::from_flat(embeddings, verse_ids, dim)
    }

    /// Build an index directly from in-memory data.
    ///
    /// Useful for tests or when embeddings have just been computed.
    pub fn from_vecs(
        embeddings: Vec<Vec<f32>>,
        verse_ids: Vec<i64>,
        dim: usize,
    ) -> Result<Self, DetectionError> {
        if embeddings.len() != verse_ids.len() {
            return Err(DetectionError::Internal(
                "embeddings and verse_ids length mismatch".into(),
            ));
        }

        for vector in &embeddings {
            if vector.len() != dim {
                return Err(DetectionError::Internal(format!(
                    "embedding dim {} != expected dim {}",
                    vector.len(),
                    dim
                )));
            }
        }

        let flat: Vec<f32> = embeddings.into_iter().flatten().collect();
        Self::from_flat(flat, verse_ids, dim)
    }

    fn from_flat(
        embeddings: Vec<f32>,
        verse_ids: Vec<i64>,
        dim: usize,
    ) -> Result<Self, DetectionError> {
        if dim == 0 {
            return Err(DetectionError::Internal(
                "vector dimension must be greater than zero".into(),
            ));
        }

        if embeddings.len() % dim != 0 {
            return Err(DetectionError::Internal(format!(
                "embeddings length {} is not a multiple of dim {}",
                embeddings.len(),
                dim
            )));
        }

        let num_vectors = embeddings.len() / dim;
        if verse_ids.len() != num_vectors {
            return Err(DetectionError::Internal(format!(
                "vector count mismatch: {} embeddings vs {} ids",
                num_vectors,
                verse_ids.len()
            )));
        }

        let hnsw = Hnsw::new(
            HNSW_M,
            num_vectors.max(1),
            HNSW_MAX_LAYER,
            HNSW_EF_CONSTRUCTION,
            DistDot {},
        );

        let mut seen_verse_ids = std::collections::HashSet::with_capacity(verse_ids.len());
        for (idx, &verse_id) in verse_ids.iter().enumerate() {
            if !seen_verse_ids.insert(verse_id) {
                return Err(DetectionError::Internal(format!(
                    "duplicate verse id in vector index: {verse_id}"
                )));
            }

            let start = idx * dim;
            let end = start + dim;
            hnsw.insert_slice((&embeddings[start..end], idx));
        }

        Ok(Self {
            hnsw,
            embeddings,
            verse_ids,
            dimension: dim,
        })
    }

    fn exact_similarity(&self, query: &[f32], row_idx: usize) -> f64 {
        let start = row_idx * self.dimension;
        let end = start + self.dimension;
        let stored = &self.embeddings[start..end];

        query
            .iter()
            .zip(stored.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>() as f64
    }
}

#[cfg(feature = "vector-search")]
impl VectorIndex for HnswVectorIndex {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, DetectionError> {
        if query.len() != self.dimension {
            return Err(DetectionError::Internal(format!(
                "query dim {} != index dim {}",
                query.len(),
                self.dimension
            )));
        }

        let n = self.verse_ids.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let candidate_count = (k * RERANK_MULTIPLIER).clamp(k, n);
        let ef_search = HNSW_EF_SEARCH.max(candidate_count);
        let hits = self.hnsw.search(query, candidate_count, ef_search);

        let mut scores: Vec<SearchResult> = Vec::with_capacity(hits.len());
        for hit in hits {
            let idx = hit.d_id;
            let Some(&verse_id) = self.verse_ids.get(idx) else {
                log::warn!("HNSW returned unknown row index {}", idx);
                continue;
            };
            scores.push(SearchResult {
                verse_id,
                similarity: self.exact_similarity(query, idx),
            });
        }

        scores.sort_unstable_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scores.truncate(k);
        Ok(scores)
    }

    fn len(&self) -> usize {
        self.verse_ids.len()
    }
}

#[cfg(all(test, feature = "vector-search"))]
mod tests {
    use super::*;

    fn make_unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn test_hnsw_search_with_exact_rerank() {
        let dim = 4;
        let embeddings = vec![
            make_unit_vec(dim, 0), // id 10
            make_unit_vec(dim, 1), // id 20
            make_unit_vec(dim, 2), // id 30
        ];
        let ids = vec![10, 20, 30];

        let index = HnswVectorIndex::from_vecs(embeddings, ids, dim).unwrap();
        assert_eq!(index.len(), 3);

        // Query closest to the second vector
        let query = make_unit_vec(dim, 1);
        let results = index.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].verse_id, 20);
        assert!((results[0].similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_duplicate_ids_rejected() {
        let dim = 2;
        let embeddings = vec![make_unit_vec(dim, 0), make_unit_vec(dim, 1)];
        let ids = vec![10, 10];
        let err = HnswVectorIndex::from_vecs(embeddings, ids, dim);
        assert!(err.is_err());
    }

    #[test]
    fn test_empty_index() {
        let index = HnswVectorIndex::from_vecs(vec![], vec![], 4).unwrap();
        assert!(index.is_empty());
        let results = index.search(&[0.0; 4], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let index = HnswVectorIndex::from_vecs(vec![vec![1.0, 0.0]], vec![1], 2).unwrap();
        let err = index.search(&[1.0, 0.0, 0.0], 1);
        assert!(err.is_err());
    }
}
