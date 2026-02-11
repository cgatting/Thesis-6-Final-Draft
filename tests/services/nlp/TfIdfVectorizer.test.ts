import { describe, it, expect } from 'vitest';
import { TfIdfVectorizer, cosineSimilarity } from '../../../src/services/nlp/TfIdfVectorizer';

describe('TfIdfVectorizer', () => {
  it('should correctly fit and transform documents', () => {
    const docs = [
      'machine learning is great',
      'machine learning is hard',
      'apples are fruit'
    ];
    
    const vectorizer = new TfIdfVectorizer();
    vectorizer.fit(docs);
    
    expect(vectorizer.vocabSize).toBeGreaterThan(0);
    
    const vec1 = vectorizer.transform('machine learning');
    const vec2 = vectorizer.transform('apples');
    
    // Check dimensions
    expect(vec1.length).toBe(vectorizer.vocabSize);
    
    // Check normalization
    const magnitude1 = vec1.reduce((acc, v) => acc + v*v, 0);
    expect(magnitude1).toBeCloseTo(1); // L2 normalized
  });

  it('should calculate cosine similarity correctly', () => {
    const v1 = [1, 0, 0];
    const v2 = [0, 1, 0];
    const v3 = [1, 1, 0]; 
    
    expect(cosineSimilarity(v1, v2)).toBe(0);
    expect(cosineSimilarity(v1, v1)).toBe(1);
    
    // Cosine of [1,0] and [1,1] is 1 / sqrt(2) ~= 0.707
    expect(cosineSimilarity(v1, v3)).toBeCloseTo(0.707, 3);
  });
});
