import { ProcessedReference, DimensionScores } from '../types';

interface OpenAlexAuthor {
  author: {
    id: string;
    display_name: string;
  };
}

interface OpenAlexWork {
  id: string;
  doi: string | null;
  title: string;
  display_name: string;
  publication_year: number;
  cited_by_count: number;
  authorships: OpenAlexAuthor[];
  primary_location?: {
    source?: {
      display_name: string;
    };
  };
  abstract_inverted_index?: Record<string, number[]>;
}

export interface OpenAlexPaper extends Omit<OpenAlexWork, 'abstract_inverted_index'> {
  abstract: string;
}

interface OpenAlexListResponse {
  meta: {
    count: number;
    db_response_time_ms: number;
    page: number;
    per_page: number;
  };
  results: OpenAlexWork[];
}

export class OpenAlexService {
  private static BASE_URL = 'https://api.openalex.org';

  /**
   * Searches for papers using OpenAlex API.
   * @param query The search query (title, keywords, etc.)
   * @param limit Number of results to return (default 3)
   */
  public async searchPapers(query: string, limit: number = 3): Promise<ProcessedReference[]> {
    try {
      // Use 'search' parameter for full-text search on works
      const url = `${OpenAlexService.BASE_URL}/works?search=${encodeURIComponent(query)}&per_page=${limit}`;
      
      console.log(`Querying OpenAlex: ${url}`);
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`OpenAlex API Error: ${response.status} ${response.statusText}`);
      }

      const data: OpenAlexListResponse = await response.json();
      
      if (!data.results || data.results.length === 0) {
        return [];
      }

      return data.results.map(work => {
        const paper = this.convertToPaper(work);
        return this.mapOpenAlexPaperToReference(paper);
      });

    } catch (error) {
      console.error("OpenAlex Search Failed:", error);
      return [];
    }
  }

  /**
   * Fetches details for multiple papers in a single batch request.
   * Uses DOIs.
   * @param ids Array of DOIs
   */
  public async getBatchDetails(ids: string[]): Promise<Record<string, OpenAlexPaper>> {
    if (ids.length === 0) return {};

    try {
        console.log(`Batch fetching metadata for ${ids.length} papers from OpenAlex...`);
        
        // OpenAlex allows filtering by multiple DOIs separated by pipe '|'
        // Clean DOIs: remove "doi:" prefix if present, just in case.
        const cleanIds = ids.map(id => id.replace(/^doi:/i, '').replace(/^https?:\/\/doi\.org\//i, ''));
        const filter = cleanIds.join('|');
        
        const url = `${OpenAlexService.BASE_URL}/works?filter=doi:${filter}&per_page=${ids.length}`;
        
        const response = await fetch(url);

        if (!response.ok) {
             throw new Error(`OpenAlex Batch API Error: ${response.status} ${response.statusText}`);
        }

        const data: OpenAlexListResponse = await response.json();
        
        const result: Record<string, OpenAlexPaper> = {};
        
        data.results.forEach((work) => {
            const paper = this.convertToPaper(work);
            
            if (paper.doi) {
                // Normalize DOI from work to match input keys if possible
                const shortDoi = paper.doi.replace('https://doi.org/', '');
                
                // We try to match with input IDs
                const originalId = ids.find(id => id.includes(shortDoi) || shortDoi.includes(id));
                if (originalId) {
                    result[originalId] = paper;
                } else {
                    result[shortDoi] = paper;
                }
            }
        });

        return result;

    } catch (error) {
        console.error("OpenAlex Batch Fetch Failed:", error);
        return {};
    }
  }

  private convertToPaper(work: OpenAlexWork): OpenAlexPaper {
      const abstract = this.reconstructAbstract(work.abstract_inverted_index);
      const { abstract_inverted_index, ...rest } = work;
      return {
          ...rest,
          abstract
      };
  }

  private mapOpenAlexPaperToReference(paper: OpenAlexPaper): ProcessedReference {
    const authors = paper.authorships?.map(a => a.author.display_name) || ["Unknown"];
    const year = paper.publication_year || new Date().getFullYear();
    
    // Generate a citation key
    const firstAuthor = authors[0]?.split(' ').pop()?.replace(/\W/g, '') || "Unknown";
    const citationKey = `${firstAuthor}${year}`;

    const scores = this.calculateScores(paper);

    return {
      id: citationKey,
      title: paper.display_name || paper.title,
      authors: authors,
      year: year,
      venue: paper.primary_location?.source?.display_name || "Unknown Venue",
      abstract: paper.abstract || "No abstract available.",
      doi: paper.doi ? paper.doi.replace('https://doi.org/', '') : undefined,
      scores: scores
    };
  }

  private reconstructAbstract(invertedIndex: Record<string, number[]> | undefined): string {
    if (!invertedIndex) return "";
    
    const words: { word: string; index: number }[] = [];
    
    Object.entries(invertedIndex).forEach(([word, indices]) => {
      indices.forEach(index => {
        words.push({ word, index });
      });
    });
    
    words.sort((a, b) => a.index - b.index);
    
    return words.map(w => w.word).join(' ');
  }

  public calculateScores(paper: OpenAlexPaper): DimensionScores {
    const recencyScore = this.calculateRecencyScore(paper.publication_year);
    const authorityScore = this.calculateAuthorityScore(paper.cited_by_count || 0);
    
    return {
        Alignment: 85, // Placeholder
        Numbers: 50,   // Placeholder
        Entities: 60,  // Placeholder
        Methods: 70,   // Placeholder
        Recency: recencyScore,
        Authority: authorityScore
    };
  }

  private calculateRecencyScore(year: number): number {
    if (!year) return 50;
    const currentYear = new Date().getFullYear();
    const age = currentYear - year;
    if (age <= 2) return 100;
    if (age <= 5) return 90;
    if (age <= 10) return 70;
    return 50;
  }

  private calculateAuthorityScore(citations: number): number {
    if (citations > 1000) return 100;
    if (citations > 100) return 90;
    if (citations > 50) return 80;
    if (citations > 10) return 70;
    return 60;
  }
}
