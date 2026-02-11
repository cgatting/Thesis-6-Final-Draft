import { ProcessedReference } from '../../types';
import { Result, ok, err } from '../../utils/result';
import { ParsingError } from '../../utils/AppError';

export class BibTexParser {
  /**
   * Parses BibTeX content into structured references.
   * @param content Raw BibTeX string
   */
  public parse(content: string): Result<ProcessedReference[], ParsingError> {
    try {
      const references: ProcessedReference[] = [];
      // Improved Regex for BibTeX entries
      // Handles @type{ key, field={val}, ... }
      const bibEntryRegex = /@(\w+)\s*\{\s*([^,]+),([^@]+)/g;
      
      let match;
      while ((match = bibEntryRegex.exec(content)) !== null) {
        const [_, type, key, body] = match;
        
        try {
          references.push({
            id: key.trim(),
            title: this.extractField(body, 'title') || `Reference ${key}`,
            authors: this.parseAuthors(this.extractField(body, 'author')),
            year: this.parseYear(this.extractField(body, 'year')),
            abstract: this.extractField(body, 'abstract') || this.extractField(body, 'title') || 'No abstract available.',
            venue: this.extractField(body, 'journal') || this.extractField(body, 'booktitle') || this.extractField(body, 'publisher'),
            doi: this.extractField(body, 'doi'),
          });
        } catch (e) {
          console.warn(`Failed to parse entry ${key}`, e);
          // Continue parsing other entries even if one fails
        }
      }

      if (references.length === 0) {
        // Fallback to simple line parser if strictly BibTeX fails
        return this.parseSimpleLines(content);
      }

      return ok(references);
    } catch (error) {
      return err(new ParsingError('Unexpected error during BibTeX parsing', error));
    }
  }

  private extractField(content: string, fieldName: string): string {
    // 1. Try { ... }
    // Note: This regex is non-recursive and will fail on nested braces like {Title with {Braces}}
    // But it's robust for standard flat BibTeX.
    const braceRegex = new RegExp(`${fieldName}\\s*=\\s*\\{([\\s\\S]*?)\\}`, 'i');
    const braceMatch = content.match(braceRegex);
    if (braceMatch) return braceMatch[1].replace(/[\n\r\s]+/g, ' ').trim();

    // 2. Try " ... "
    const quoteRegex = new RegExp(`${fieldName}\\s*=\\s*"([\\s\\S]*?)"`, 'i');
    const quoteMatch = content.match(quoteRegex);
    if (quoteMatch) return quoteMatch[1].replace(/[\n\r\s]+/g, ' ').trim();
    
    // 3. Fallback
    const simpleRegex = new RegExp(`${fieldName}\\s*=\\s*([^,}\n]+)`, 'i');
    const simpleMatch = content.match(simpleRegex);
    return simpleMatch ? simpleMatch[1].trim() : '';
  }

  private parseAuthors(authorStr: string): string[] {
    if (!authorStr) return ['Unknown'];
    return authorStr.split(/\s+and\s+/i).map(a => {
        // Normalize "Last, First" to "First Last" if needed, or keep as is.
        // For now, just trim.
        return a.trim();
    });
  }

  private parseYear(yearStr: string): number {
    if (!yearStr) return new Date().getFullYear();
    const match = yearStr.match(/\d{4}/);
    return match ? parseInt(match[0], 10) : new Date().getFullYear();
  }

  private parseSimpleLines(content: string): Result<ProcessedReference[], ParsingError> {
    const references: ProcessedReference[] = [];
    const lines = content.split('\n').filter(l => l.trim().length > 10);
    
    lines.forEach((line, index) => {
      // Heuristic: [1] Author (Year). Title.
      const yearMatch = line.match(/\((\d{4})\)/) || line.match(/\b(19|20)\d{2}\b/);
      const year = yearMatch ? parseInt(yearMatch[1] || yearMatch[0], 10) : new Date().getFullYear();
      const id = `ref_${index + 1}`;
      
      references.push({
        id,
        title: line.substring(0, 100) + "...",
        authors: ["Unknown"],
        year,
        abstract: line,
      });
    });
    
    if (references.length === 0) {
        return err(new ParsingError('Could not parse any references from text'));
    }

    return ok(references);
  }
}
