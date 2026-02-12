export interface DimensionScores {
  Alignment: number; // 45%
  Numbers: number;   // 20%
  Entities: number;  // 15%
  Methods: number;   // 10%
  Recency: number;   // 7%
  Authority: number; // 3%
}

export interface ProcessedReference {
  id: string; // The citation key (e.g., Author2023)
  title: string;
  abstract: string;
  authors: string[];
  year: number;
  venue?: string;
  doi?: string;
  citationCount?: number;
  embedding?: number[]; // Vector representation
  scores?: DimensionScores;
  relevanceSummary?: string;
}

export interface AnalyzedSentence {
  text: string;
  citations: string[]; // Keys found in this sentence
  entities: string[];  // Extracted entities
  hasNumbers: boolean; // Does the sentence contain numerical claims?
  embedding?: number[];
  scores?: Record<string, DimensionScores>; // Map citation key to scores
  
  // Advanced Analysis Fields
  isMissingCitation?: boolean;
  isHighImpact?: boolean;
  gapIdentified?: boolean;
  triggerPhrase?: string;
  analysisNotes?: string[];
  suggestedReferences?: ProcessedReference[];
}

export interface AnalysisResult {
  overallScore: number;
  analyzedSentences: AnalyzedSentence[];
  references: Record<string, ProcessedReference>;
  summary: string;
  documentTitle?: string;
  dimensionScores: DimensionScores;
  gaps: string[];
}

export enum AppState {
  IDLE = 'IDLE',
  PARSING = 'PARSING',
  EMBEDDING = 'EMBEDDING',
  SCORING = 'SCORING',
  RESULTS = 'RESULTS',
  ERROR = 'ERROR'
}

export type FileType = 'manuscript' | 'bibliography';