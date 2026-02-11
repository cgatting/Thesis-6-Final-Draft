import { tokenize } from './tokenizer';

export class TfIdfVectorizer {
  private vocabulary: Map<string, number> = new Map();
  private idf: Map<string, number> = new Map();
  
  public get vocabSize(): number {
    return this.vocabulary.size;
  }

  /**
   * Fit the model to a corpus of documents to learn vocabulary and IDF.
   * @param documents Array of text documents
   */
  fit(documents: string[]): void {
    this.vocabulary.clear();
    this.idf.clear();

    const docCount = documents.length;
    const termDocCounts: Map<string, number> = new Map();
    
    // 1. Build Vocabulary and count document frequency
    documents.forEach(doc => {
      const tokens = new Set(tokenize(doc)); // Set for unique words in doc to count DF
      tokens.forEach(token => {
        termDocCounts.set(token, (termDocCounts.get(token) || 0) + 1);
        if (!this.vocabulary.has(token)) {
          this.vocabulary.set(token, this.vocabulary.size);
        }
      });
    });

    // 2. Compute IDF: log((N + 1) / (df + 1)) + 1 (sklearn style smooth_idf)
    // We iterate over all terms in vocabulary. If a term is in vocab but not in termDocCounts 
    // (which shouldn't happen here as we build vocab from docs), it would be handled.
    this.vocabulary.forEach((_, term) => {
        const df = termDocCounts.get(term) || 0;
        const idfValue = Math.log((docCount + 1) / (df + 1)) + 1;
        this.idf.set(term, idfValue);
    });
  }

  /**
   * Transform a document to a TF-IDF vector.
   * @param text Input text
   * @returns Dense vector as array of numbers
   */
  transform(text: string): number[] {
    const tokens = tokenize(text);
    const vector = new Array(this.vocabulary.size).fill(0);
    
    if (tokens.length === 0) return vector;

    const termCounts: Map<string, number> = new Map();

    // TF: Raw count
    tokens.forEach(t => {
      termCounts.set(t, (termCounts.get(t) || 0) + 1);
    });

    termCounts.forEach((count, term) => {
      const idx = this.vocabulary.get(term);
      if (idx !== undefined) {
        // TF-IDF = count * IDF (Standard definition often uses raw count, 
        // sometimes normalized by doc length. Let's use raw count * idf and then L2 normalize vector)
        const idf = this.idf.get(term) || 0;
        vector[idx] = count * idf;
      }
    });

    // L2 Normalization
    let norm = 0;
    for (let v of vector) norm += v * v;
    norm = Math.sqrt(norm);
    
    if (norm > 0) {
        for(let i=0; i<vector.length; i++) vector[i] /= norm;
    }

    return vector;
  }
}

export const cosineSimilarity = (vecA: number[], vecB: number[]): number => {
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
  
  let dotProduct = 0;
  // Norms are 1 if L2 normalized, but we calculate to be safe
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
};
