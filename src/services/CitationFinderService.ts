import { ProcessedReference, AnalyzedSentence } from '../types';
import { upsertAndSortBibTexEntries } from './parsers/BibTexFileEditor';
import { OpenAlexService } from './OpenAlexService';
import { TfIdfVectorizer } from './nlp/TfIdfVectorizer';
import { EntityExtractor } from './nlp/EntityExtractor';
import { ScoringEngine, computeWeightedTotal } from './scoring/ScoringEngine';

export class CitationFinderService {
  private oaService = new OpenAlexService();
  private vectorizer = new TfIdfVectorizer();
  private entityExtractor = new EntityExtractor();
  private scoringEngine = new ScoringEngine();

  /**
   * Queries OpenAlex for better alternative sources.
   * @param currentRef The reference currently being used
   * @param contextSentence The sentence where it is cited
   */
  public async findBetterSources(currentRef: ProcessedReference, contextSentence: string): Promise<ProcessedReference[]> {
    
    // Extract keywords from the sentence context
    const keywords = contextSentence
      .split(/\W+/)
      .filter(w => w.length > 5 && !['reference', 'citation', 'however', 'because', 'although'].includes(w.toLowerCase()))
      .slice(0, 3)
      .join(' ');

    const topic = keywords || currentRef.title.split(':')[0];
    
    console.log(`Querying OpenAlex for topic: "${topic}"`);

    // Fetch 15 candidates to score and filter
    const candidates = await this.oaService.searchPapers(topic, 15);

    if (candidates.length === 0) return [];

    return this.scoreAndRankCandidates(candidates, contextSentence);
  }

  /**
   * Finds papers to fill a detected research gap or missing citation.
   */
  public async findSourcesForGap(contextSentence: string): Promise<ProcessedReference[]> {
    // Extract potential search terms
    const keywords = contextSentence
      .split(/\W+/)
      .filter(w => w.length > 5 && !['however', 'because', 'although', 'studies', 'shown', 'research', 'indicated'].includes(w.toLowerCase()))
      .slice(0, 4)
      .join(' ');

    if (!keywords) return [];

    console.log(`Searching OpenAlex for gap filling: "${keywords}"`);
    
    const candidates = await this.oaService.searchPapers(keywords, 10);
    
    if (candidates.length === 0) return [];

    return this.scoreAndRankCandidates(candidates, contextSentence);
  }

  private scoreAndRankCandidates(candidates: ProcessedReference[], contextSentence: string): ProcessedReference[] {
    // 1. Prepare Corpus (Context + Candidates)
    const corpus = [
      contextSentence,
      ...candidates.map(c => c.abstract)
    ];

    // 2. Fit Vectorizer locally
    this.vectorizer.fit(corpus);

    // 3. Analyze Context Sentence
    const sentenceEmbedding = this.vectorizer.transform(contextSentence);
    const sentenceEntities = this.entityExtractor.extract(contextSentence);
    
    const analyzedSentence: AnalyzedSentence = {
      text: contextSentence,
      embedding: sentenceEmbedding,
      entities: sentenceEntities,
      hasNumbers: /\d/.test(contextSentence),
      isMissingCitation: false,
      isHighImpact: false,
      gapIdentified: false,
      citations: []
    };

    // 4. Score Candidates
    const scoredCandidates = candidates.map(candidate => {
      // Generate embedding for candidate abstract
      candidate.embedding = this.vectorizer.transform(candidate.abstract);

      // Calculate detailed scores
      const scores = this.scoringEngine.calculateScore(analyzedSentence, candidate);
      candidate.scores = scores;
      
      return {
        candidate,
        totalScore: computeWeightedTotal(scores)
      };
    });

    // 5. Sort by Total Score and return top 3
    return scoredCandidates
      .sort((a, b) => b.totalScore - a.totalScore)
      .slice(0, 3)
      .map(item => item.candidate);
  }

  /**
   * Generates a BibTeX entry for the new source
   */
  public generateBibTeX(ref: ProcessedReference): string {
    return `@article{${ref.id},
  author = {${ref.authors.join(' and ')}},
  title = {${ref.title}},
  journal = {${ref.venue}},
  year = {${ref.year}},
  abstract = {${ref.abstract}}${ref.doi ? `,\n  doi = {${ref.doi}}` : ''}
}`;
  }

  /**
   * Updates the .tex and .bib files with the new reference
   */
  public updateFiles(
    oldId: string, 
    newRef: ProcessedReference, 
    manuscriptContent: string, 
    bibContent: string
  ): { manuscript: string, bib: string } {
    
    // Robust replacement of citation key in the manuscript
    const escapedId = oldId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`\\b${escapedId}\\b`, 'g');
    
    const newManuscript = manuscriptContent.replace(regex, newRef.id);

    // Update bibliography
    const newBib = upsertAndSortBibTexEntries(bibContent, [this.generateBibTeX(newRef)]);

    return { manuscript: newManuscript, bib: newBib };
  }
}
