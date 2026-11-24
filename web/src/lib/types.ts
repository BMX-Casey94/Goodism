export interface ProvenanceRecord {
  doc_id: string;
  version: number;
  section_path: string;
  txid: string;
  vout: number;
  published_at: string | null;
  verified: boolean;
}

export interface Passage {
  doc_id: string;
  title: string;
  authority: string;
  precedence: number;
  status: string;
  caveat: string | null;
  sections: string[];
  start: number;
  end: number;
  text: string;
  score: number;
  fusion_score: number;
  weak_match: boolean;
  queries: number[];
  provenance: ProvenanceRecord[];
}

export interface ConflictRecord {
  doc_id: string;
  conflicts_with: string;
  prevails?: string;
  reason?: string;
  notes?: string;
  topic?: string;
}

export interface DocumentSummary {
  doc_id: string;
  title: string;
  domain: string;
  authority: string;
  summary: string;
  sections: number;
  topics?: string[];
}

export interface IndexStatus {
  pinned_snapshot: number;
  embedding_model: string;
  documents: number;
  chunks: number;
  created_at?: string;
  qa_enabled?: boolean;
  llm_model?: string;
}

export interface NeighbourChunk {
  ordinal: number;
  heading_path: string;
  text: string;
}

export type FeedbackRating = "accurate" | "needs_review" | "inaccurate";

export interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: "qa" | "evidence";
  model?: string;
  passages?: Passage[];
  conflicts?: ConflictRecord[];
  weakMatch?: boolean;
  streaming?: boolean;
  error?: string;
  userQuestion?: string;
}

export interface MetaPayload {
  status: IndexStatus | null;
  documents: DocumentSummary[];
  conflicts: ConflictRecord[];
  llmConfigured: boolean;
  model: string;
  backendReachable: boolean;
  backendError?: string | null;
}
