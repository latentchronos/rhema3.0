//! Utility to pre-compute embeddings for every Bible verse and persist
//! them to binary files that `HnswVectorIndex::load` can read.
//!
//! This module requires the `onnx` feature so it has access to
//! `OnnxEmbedder`.

#[cfg(feature = "onnx")]
use std::fs::OpenOptions;
#[cfg(feature = "onnx")]
use std::io::{Seek, SeekFrom, Write};
#[cfg(feature = "onnx")]
use std::path::Path;

#[cfg(feature = "onnx")]
use crate::error::DetectionError;
#[cfg(feature = "onnx")]
use super::embedder::TextEmbedder;
#[cfg(feature = "onnx")]
use super::onnx_embedder::OnnxEmbedder;

/// Number of verses already fully persisted (and safe to skip on resume). A verse
/// counts only when BOTH output files hold a complete record for it, so the resume
/// point is the min of the two files' whole-record counts, capped at `total`. Integer
/// division drops a torn final record left by a mid-write crash.
#[cfg(feature = "onnx")]
fn resume_point(emb_len: u64, ids_len: u64, emb_rec: u64, id_rec: u64, total: usize) -> usize {
    if emb_rec == 0 || id_rec == 0 {
        return 0;
    }
    let emb_done = emb_len / emb_rec;
    let ids_done = ids_len / id_rec;
    emb_done.min(ids_done).min(total as u64) as usize
}

/// Pre-compute embeddings for a set of verses and write the results to
/// binary files.
///
/// # Arguments
///
/// * `embedder` -- an `OnnxEmbedder` whose prompt prefix should be set to
///   `"passage: "` for document embedding (as opposed to `"query: "` used
///   at search time).
/// * `verses` -- `(verse_id, verse_text)` pairs.
/// * `output_embeddings_path` -- destination for the raw `f32` embedding
///   matrix.
/// * `output_ids_path` -- destination for the raw `i64` verse-ID array.
///
/// Both files are written in the platform's native byte order.  The
/// embeddings file is a flat array of `f32` values (`dim * num_verses`
/// floats) and the IDs file is a flat array of `i64` values
/// (`num_verses` entries).
#[cfg(feature = "onnx")]
pub fn precompute_embeddings(
    embedder: &OnnxEmbedder,
    verses: &[(i64, String)],
    output_embeddings_path: &Path,
    output_ids_path: &Path,
) -> Result<(), DetectionError> {
    let total = verses.len();
    let dim = embedder.dimension();
    let emb_rec = (dim * std::mem::size_of::<f32>()) as u64;
    let id_rec = std::mem::size_of::<i64>() as u64;

    // Resume: skip verses already fully written to BOTH files (a torn final record
    // from a crash is dropped). Delete the output files to force a clean rebuild.
    let emb_existing = std::fs::metadata(output_embeddings_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let ids_existing = std::fs::metadata(output_ids_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let start = resume_point(emb_existing, ids_existing, emb_rec, id_rec, total);

    log::info!(
        "Pre-computing embeddings for {} verses (starting at {}) ...",
        total, start
    );

    let (mut emb_file, mut ids_file) = if start > 0 {
        log::info!(
            "  resuming: {}/{} already done (delete {} to rebuild from scratch)",
            start,
            total,
            output_embeddings_path.display()
        );
        let mut emb = OpenOptions::new()
            .write(true)
            .open(output_embeddings_path)
            .map_err(|e| {
                DetectionError::Internal(format!(
                    "open {}: {e}",
                    output_embeddings_path.display()
                ))
            })?;
        // Drop any torn/excess tail so appends stay record-aligned and in sync.
        emb.set_len(start as u64 * emb_rec)
            .map_err(|e| DetectionError::Internal(format!("truncate embeddings: {e}")))?;
        emb.seek(SeekFrom::End(0))
            .map_err(|e| DetectionError::Internal(format!("seek embeddings: {e}")))?;

        let mut ids = OpenOptions::new()
            .write(true)
            .open(output_ids_path)
            .map_err(|e| {
                DetectionError::Internal(format!("open {}: {e}", output_ids_path.display()))
            })?;
        ids.set_len(start as u64 * id_rec)
            .map_err(|e| DetectionError::Internal(format!("truncate ids: {e}")))?;
        ids.seek(SeekFrom::End(0))
            .map_err(|e| DetectionError::Internal(format!("seek ids: {e}")))?;

        (emb, ids)
    } else {
        let emb = std::fs::File::create(output_embeddings_path).map_err(|e| {
            DetectionError::Internal(format!(
                "create {}: {e}",
                output_embeddings_path.display()
            ))
        })?;
        let ids = std::fs::File::create(output_ids_path).map_err(|e| {
            DetectionError::Internal(format!("create {}: {e}", output_ids_path.display()))
        })?;
        (emb, ids)
    };

    // Embed in batches to amortize per-call ONNX overhead (~10x fewer runs than
    // one-verse-at-a-time). Each batch row is identical to a single `embed` (proven
    // by the embedder's equivalence test), and rows are written in verse order, so
    // the on-disk byte layout is unchanged. Tunable via RHEMA_PRECOMPUTE_BATCH.
    let batch_size: usize = std::env::var("RHEMA_PRECOMPUTE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(32);
    log::info!("  (batch size {batch_size})");

    let mut done = start;
    for chunk in verses[start..].chunks(batch_size) {
        let texts: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
        let embeddings = embedder.embed_batch(&texts)?;

        for ((verse_id, _), embedding) in chunk.iter().zip(embeddings.iter()) {
            // Write f32 vector as raw bytes (native byte order).
            // Safety: f32 has no padding and a well-defined repr.
            let emb_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    embedding.as_ptr() as *const u8,
                    embedding.len() * std::mem::size_of::<f32>(),
                )
            };
            emb_file.write_all(emb_bytes).map_err(|e| {
                DetectionError::Internal(format!("write embedding: {e}"))
            })?;

            // Write verse_id as raw i64 bytes (native byte order).
            let id_bytes = verse_id.to_ne_bytes();
            ids_file.write_all(&id_bytes).map_err(|e| {
                DetectionError::Internal(format!("write id: {e}"))
            })?;
        }

        let prev = done;
        done += chunk.len();
        // Log once per ~1000 verses crossed (and at the end).
        if done == total || done / 1000 != prev / 1000 {
            log::info!("  embedded {}/{} verses", done, total);
        }
    }

    log::info!("Pre-computation complete. Files written.");
    Ok(())
}

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::resume_point;

    const EMB: u64 = 1024 * 4; // dim 1024, f32
    const ID: u64 = 8; // i64

    #[test]
    fn fresh_files_start_at_zero() {
        assert_eq!(resume_point(0, 0, EMB, ID, 31102), 0);
    }

    #[test]
    fn counts_whole_records() {
        assert_eq!(resume_point(5 * EMB, 5 * ID, EMB, ID, 31102), 5);
    }

    #[test]
    fn drops_a_torn_final_record() {
        // 5 complete embeddings plus a half-written 6th (crash mid-write).
        assert_eq!(resume_point(5 * EMB + 100, 5 * ID, EMB, ID, 31102), 5);
    }

    #[test]
    fn uses_the_min_of_mismatched_files() {
        // Embeddings raced ahead of ids before the crash → resume at the lesser.
        assert_eq!(resume_point(5 * EMB, 3 * ID, EMB, ID, 31102), 3);
    }

    #[test]
    fn caps_at_total() {
        assert_eq!(resume_point(100 * EMB, 100 * ID, EMB, ID, 50), 50);
    }

    #[test]
    fn zero_record_size_is_safe() {
        assert_eq!(resume_point(EMB, ID, 0, ID, 10), 0);
    }
}
