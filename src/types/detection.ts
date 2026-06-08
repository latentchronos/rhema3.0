export interface DetectionResult {
  verse_ref: string
  verse_text: string
  book_name: string
  book_number: number
  chapter: number
  verse: number
  confidence: number
  source:
    | "direct"
    | "contextual"
    | "quotation"
    | "semantic_local"
    | "semantic_cloud"
  auto_queued: boolean
  raw_score: number
  minimum_threshold: number
  auto_queue_threshold: number
  decision: "auto_queued" | "review_required"
  explanation: string
  transcript_snippet: string
}

export interface DetectionStatus {
  has_direct: boolean
  has_semantic: boolean
  has_cloud: boolean
}

export interface SemanticSearchResult {
  verse_ref: string
  verse_text: string
  book_name: string
  book_number: number
  chapter: number
  verse: number
  similarity: number
}
