import { DimensionScores, AnalyzedSentence, ProcessedReference } from '../../types';
import { cosineSimilarity } from '../nlp/TfIdfVectorizer';

export interface ScoringConfig {
  weights: {
    Alignment: number;
    Numbers: number;
    Entities: number;
    Methods: number;
    Recency: number;
    Authority: number;
  };
}

export class ScoringEngine {
  constructor(private config: ScoringConfig = ScoringEngine.DEFAULT_CONFIG) {}

  public static DEFAULT_CONFIG: ScoringConfig = {
    weights: {
      Alignment: 0.30,
      Numbers: 0.10,
      Entities: 0.20,
      Methods: 0.15,
      Recency: 0.10,
      Authority: 0.15
    }
  };

  public calculateScore(sentence: AnalyzedSentence, reference: ProcessedReference): DimensionScores {
    // 1. Alignment
    let alignmentScore = 0;
    if (sentence.embedding && reference.embedding) {
        const rawSim = cosineSimilarity(sentence.embedding, reference.embedding);
        alignmentScore = Math.max(0, rawSim * 100);
    }

    // 2. Numbers
    const numberScore = this.calculateNumberScore(sentence.hasNumbers, reference.abstract);

    // 3. Entities
    const entityScore = this.calculateEntityScore(sentence.entities, reference.abstract);

    // 4. Methods
    const methodScore = /method|approach|algorithm|framework|metric/i.test(reference.abstract) ? 90 : 50;

    // 5. Recency
    const recencyScore = reference.scores?.Recency ?? this.calculateRecencyScore(reference.year);

    // 6. Authority
    const authorityScore = reference.scores?.Authority ?? 60; // Default lower if unknown

    return {
      Alignment: alignmentScore,
      Numbers: numberScore,
      Entities: entityScore,
      Methods: methodScore,
      Recency: recencyScore,
      Authority: authorityScore
    };
  }

  public computeWeightedTotal(scores: DimensionScores): number {
    const w = this.config.weights;
    return (
      (scores.Alignment * w.Alignment) +
      (scores.Numbers * w.Numbers) +
      (scores.Entities * w.Entities) +
      (scores.Methods * w.Methods) +
      (scores.Recency * w.Recency) +
      (scores.Authority * w.Authority)
    );
  }

  private calculateEntityScore(sentenceEntities: string[], refAbstract: string): number {
    if (sentenceEntities.length === 0) return 50; 
    
    const abstractLower = refAbstract.toLowerCase();
    let matches = 0;
    
    sentenceEntities.forEach(entity => {
      if (abstractLower.includes(entity.toLowerCase())) {
        matches++;
      }
    });

    return Math.min(100, (matches / sentenceEntities.length) * 100);
  }

  private calculateRecencyScore(year: number): number {
    const currentYear = new Date().getFullYear();
    const age = currentYear - year;
    
    if (age <= 2) return 100;
    if (age <= 5) return 85;
    if (age <= 10) return 60;
    if (age <= 20) return 40;
    return 20;
  }

  private calculateNumberScore(hasNumbers: boolean, refAbstract: string): number {
    if (!hasNumbers) return 100;
    const abstractHasNumbers = /\d/.test(refAbstract);
    return abstractHasNumbers ? 100 : 30;
  }
}

// Standalone export for UI
export const computeWeightedTotal = (scores: DimensionScores): number => {
    return new ScoringEngine().computeWeightedTotal(scores);
};
