import { AnalysisResult, AnalyzedSentence, ProcessedReference, DimensionScores } from '../types';
import { TfIdfVectorizer } from './nlp/TfIdfVectorizer';
import { EntityExtractor } from './nlp/EntityExtractor';
import { BibTexParser } from './parsers/BibTexParser';
import { LatexParser } from './parsers/LatexParser';
import { ScoringEngine } from './scoring/ScoringEngine';
import { OpenAlexService } from './OpenAlexService';
import { CitationFinderService } from './CitationFinderService';
import { unwrap } from '../utils/result';

export class AnalysisService {
  private vectorizer = new TfIdfVectorizer();
  private entityExtractor = new EntityExtractor();
  private bibParser = new BibTexParser();
  private latexParser = new LatexParser();
  private oaService = new OpenAlexService();
  private citationFinder: CitationFinderService;
  private scoringEngine: ScoringEngine;

  constructor(config?: any) {
    this.scoringEngine = new ScoringEngine(config);
    this.citationFinder = new CitationFinderService(config);
  }

  public async analyze(manuscriptText: string, bibText: string): Promise<AnalysisResult> {
    console.log('Starting analysis...');

    // 1. Parse Inputs
    const referencesResult = this.bibParser.parse(bibText);
    const references = unwrap(referencesResult); // Will throw if parsing fails
    
    const sentencesResult = this.latexParser.parse(manuscriptText);
    const sentences = unwrap(sentencesResult);

    const documentTitle = this.latexParser.extractTitle(manuscriptText);

    console.log(`Parsed ${references.length} references and ${sentences.length} sentences.`);

    // 1b. Batch Fetch Metadata from OpenAlex
    const dois = references
      .map(r => r.doi)
      .filter((d): d is string => !!d);

    if (dois.length > 0) {
      console.log(`Attempting to batch fetch metadata for ${dois.length} references...`);
      const oaData = await this.oaService.getBatchDetails(dois);
      
      // Enrich references with OpenAlex data
      references.forEach(ref => {
        if (ref.doi && oaData[ref.doi]) {
          const oaPaper = oaData[ref.doi];
          const oaScores = this.oaService.calculateScores(oaPaper);
          
          // Update Abstract if missing or short
          if ((!ref.abstract || ref.abstract.length < 50) && oaPaper.abstract) {
             ref.abstract = oaPaper.abstract;
          }
          
          // Update Venue if unknown
          const venue = oaPaper.primary_location?.source?.display_name;
          if ((!ref.venue || ref.venue === 'Unknown') && venue) {
             ref.venue = venue;
          }

          // Pre-populate scores with real data
          ref.scores = oaScores;
          
          // Update citation count
          if (oaPaper.cited_by_count !== undefined) {
             ref.citationCount = oaPaper.cited_by_count;
          }
        }
      });
    }

    // 2. Prepare Corpus for Vectorizer
    // Corpus = All abstracts + All manuscript sentences
    const corpus: string[] = [
      ...references.map(r => r.abstract),
      ...sentences.map(s => s.text)
    ];

    // 3. Fit Vectorizer
    this.vectorizer.fit(corpus);
    console.log(`Vectorizer fitted with vocab size: ${this.vectorizer.vocabSize}`);

    // 4. Process References (Embeddings)
    const processedReferences: Record<string, ProcessedReference> = {};
    references.forEach(ref => {
      ref.embedding = this.vectorizer.transform(ref.abstract);
      processedReferences[ref.id] = ref;
    });

    // 5. Process Sentences (Embeddings + Entities + Scoring)
    let totalScore = 0;
    let scoredSentencesCount = 0;
    
    // Helper to accumulate scores for references
    const referenceScoreAccumulator: Record<string, DimensionScores[]> = {};

    const analyzedSentences = await Promise.all(sentences.map(async sentence => {
      // a. Embedding
      sentence.embedding = this.vectorizer.transform(sentence.text);
      
      // b. Entities
      sentence.entities = this.entityExtractor.extract(sentence.text);

      // c. Scoring against cited references
      const scoresMap: Record<string, DimensionScores> = {};
      
      if (sentence.citations && sentence.citations.length > 0) {
        sentence.citations.forEach(citeKey => {
          const ref = processedReferences[citeKey];
          if (ref) {
            const scores = this.scoringEngine.calculateScore(sentence, ref);
            scoresMap[citeKey] = scores;
            
            // Accumulate for reference-level aggregation
            if (!referenceScoreAccumulator[citeKey]) {
                referenceScoreAccumulator[citeKey] = [];
            }
            referenceScoreAccumulator[citeKey].push(scores);

            // Add to total score average (simplified aggregation)
            totalScore += this.scoringEngine.computeWeightedTotal(scores);
            scoredSentencesCount++;
          }
        });
      }

      sentence.scores = scoresMap;
      
      // d. Advanced Analysis Checks
      const missingCitationTrigger = this.detectMissingCitation(sentence);
      const highImpactTrigger = this.detectHighImpact(sentence);
      const gapTrigger = this.detectGap(sentence);

      sentence.isMissingCitation = !!missingCitationTrigger;
      sentence.isHighImpact = !!highImpactTrigger;
      sentence.gapIdentified = !!gapTrigger;
      sentence.triggerPhrase = missingCitationTrigger || highImpactTrigger || gapTrigger || undefined;
      sentence.analysisNotes = [];

      if (missingCitationTrigger) {
        sentence.analysisNotes.push(`Flagged as a claim requiring evidence due to the phrase "${missingCitationTrigger}".`);
        sentence.analysisNotes.push("Consider adding a citation or clarifying if this is your own finding.");
        
        // Smart Gap Filling: Suggest papers
        const suggestions = await this.citationFinder.findSourcesForGap(sentence.text);
        if (suggestions.length > 0) {
            sentence.suggestedReferences = suggestions;
            sentence.analysisNotes.push(`Found ${suggestions.length} potential sources to support this claim.`);
        }
      }
      if (highImpactTrigger) {
        sentence.analysisNotes.push(`High-impact statement detected via "${highImpactTrigger}".`);
      }
      if (gapTrigger) {
        sentence.analysisNotes.push(`Identifies a potential research gap using "${gapTrigger}".`);
        
        // Smart Gap Filling: Suggest papers for the gap
        const suggestions = await this.citationFinder.findSourcesForGap(sentence.text);
        if (suggestions.length > 0) {
            sentence.suggestedReferences = suggestions;
            sentence.analysisNotes.push(`Found ${suggestions.length} relevant papers for this topic.`);
        }
      }

      return sentence;
    }));

    // 5.5 Aggregate Scores back to References
    Object.keys(referenceScoreAccumulator).forEach(refId => {
        const scoresList = referenceScoreAccumulator[refId];
        if (scoresList.length > 0) {
            const avgScores: DimensionScores = {
                Alignment: 0, Numbers: 0, Entities: 0, Methods: 0, Recency: 0, Authority: 0
            };
            
            scoresList.forEach(s => {
                avgScores.Alignment += s.Alignment;
                avgScores.Numbers += s.Numbers;
                avgScores.Entities += s.Entities;
                avgScores.Methods += s.Methods;
                avgScores.Recency += s.Recency;
                avgScores.Authority += s.Authority;
            });

            // Calculate averages
            const count = scoresList.length;
            avgScores.Alignment /= count;
            avgScores.Numbers /= count;
            avgScores.Entities /= count;
            avgScores.Methods /= count;
            avgScores.Recency /= count;
            avgScores.Authority /= count;

            if (processedReferences[refId]) {
                processedReferences[refId].scores = avgScores;
            }
        }
    });

    // 6. Calculate Overall Score
    const finalScore = scoredSentencesCount > 0 ? Number((totalScore / scoredSentencesCount).toFixed(2)) : 0;

    // 7. Generate Summary (Stub)
    const summary = `Analyzed ${sentences.length} sentences against ${references.length} references. Overall alignment score: ${finalScore}/100.`;

    // 8. Identify Gaps from Analyzed Sentences
    const gaps = analyzedSentences
      .filter(s => s.gapIdentified || s.isMissingCitation)
      .map(s => s.triggerPhrase ? `Gap detected via "${s.triggerPhrase}" in: "${s.text.substring(0, 50)}..."` : `Gap in: "${s.text.substring(0, 50)}..."`);
    
    if (gaps.length === 0) {
        gaps.push("No major gaps detected.");
    }

    // 9. Aggregate Dimension Scores (Average across all citations)
    // This is a bit rough, but suffices for now
    const avgDimensions: DimensionScores = {
        Alignment: 0, Numbers: 0, Entities: 0, Methods: 0, Recency: 0, Authority: 0
    };

    if (scoredSentencesCount > 0) {
        analyzedSentences.forEach(s => {
            if (s.scores) {
                Object.values(s.scores).forEach(score => {
                    avgDimensions.Alignment += score.Alignment;
                    avgDimensions.Numbers += score.Numbers;
                    avgDimensions.Entities += score.Entities;
                    avgDimensions.Methods += score.Methods;
                    avgDimensions.Recency += score.Recency;
                    avgDimensions.Authority += score.Authority;
                });
            }
        });
        const count = scoredSentencesCount; // or total number of citation pairs
        // Wait, scoredSentencesCount is actually total citation pairs above?
        // Let's re-verify logic.
        // Yes, totalScore += ... happens per citation.
        // So count is correct divisor.
        
        avgDimensions.Alignment = avgDimensions.Alignment / count;
        avgDimensions.Numbers = avgDimensions.Numbers / count;
        avgDimensions.Entities = avgDimensions.Entities / count;
        avgDimensions.Methods = avgDimensions.Methods / count;
        avgDimensions.Recency = avgDimensions.Recency / count;
        avgDimensions.Authority = avgDimensions.Authority / count;
    }

    return {
      overallScore: finalScore,
      analyzedSentences,
      references: processedReferences,
      summary,
      documentTitle,
      dimensionScores: avgDimensions,
      gaps
    };
  }

  private detectMissingCitation(sentence: AnalyzedSentence): string | null {
    // 1. If it already has citations, it's not missing one.
    if (sentence.citations && sentence.citations.length > 0) return null;

    // 2. Ignore short sentences or titles (heuristic)
    if (sentence.text.length < 50) return null;

    // 3. Look for claim markers
    const claimMarkers = [
      "shown that", "demonstrated that", "observed that", "found that", 
      "results indicate", "studies have", "previous work", "established that",
      "known to be", "evidence suggests", "data shows", "concluded that"
    ];

    const lowerText = sentence.text.toLowerCase();
    const match = claimMarkers.find(marker => lowerText.includes(marker));
    return match || null;
  }

  private detectHighImpact(sentence: AnalyzedSentence): string | null {
    const impactMarkers = [
      "significant", "novel", "critical", "breakthrough", "key finding", 
      "important", "essential", "major contribution", "first time",
      "demonstrates for the first time", "crucial", "fundamental"
    ];

    const lowerText = sentence.text.toLowerCase();
    const match = impactMarkers.find(marker => lowerText.includes(marker));
    return match || null;
  }

  private detectGap(sentence: AnalyzedSentence): string | null {
    const gapMarkers = [
      "unknown", "unclear", "limited", "lack of", "remains to be", 
      "future work", "little is known", "not well understood",
      "gap in the literature", "needs further", "unresolved"
    ];

    const lowerText = sentence.text.toLowerCase();
    const match = gapMarkers.find(marker => lowerText.includes(marker));
    return match || null;
  }
}
