import { AnalyzedSentence } from '../../types';
import { Result, ok, err } from '../../utils/result';
import { ParsingError } from '../../utils/AppError';

export class LatexParser {
  private static CITE_EXTRACT_REGEX = /(\\(?:cite|parencite|textcite|footcite)[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\})/g;

  public extractTitle(content: string): string | undefined {
    const titleStart = content.indexOf('\\title{');
    if (titleStart === -1) return undefined;

    let depth = 0;
    let start = titleStart + 7; // Length of "\title{"
    let end = -1;

    for (let i = start; i < content.length; i++) {
      if (content[i] === '{') {
        depth++;
      } else if (content[i] === '}') {
        if (depth === 0) {
          end = i;
          break;
        }
        depth--;
      }
    }

    if (end !== -1) {
      const rawContent = content.substring(start, end);
      // Clean first to handle nested formatting like \textbf{Title \\ Subtitle}
      const cleaned = this.cleanLatex(rawContent);
      // Split by double backslash (\\) or \newline command and take the first part
      // Note: cleanLatex preserves \\ and \newline but replaces \n with space
      const parts = cleaned.split(/\\\\|\\newline/);
      return parts[0].trim();
    }
    return undefined;
  }

  /**
   * Removes content that should not be parsed (verbatim, comment environments, iffalse blocks)
   */
  private removeNonTextContent(content: string): string {
    return content
      .replace(/\\begin\{verbatim\}([\s\S]*?)\\end\{verbatim\}/g, '')
      .replace(/\\begin\{comment\}([\s\S]*?)\\end\{comment\}/g, '')
      .replace(/\\iffalse([\s\S]*?)\\fi/g, '');
  }

  /**
   * Extracts all citation keys from the LaTeX content.
   * Handles \cite{key}, \cite{key1, key2}, \parencite{key}, etc.
   */
  public extractCitations(content: string): string[] {
    // Clean content first to avoid extracting citations from ignored blocks
    let cleanContent = this.removeNonTextContent(content);
    // Also remove comments for simple extraction
    cleanContent = cleanContent.replace(/%.*$/gm, '');

    const citations = new Set<string>();
    // Regex to match citation commands and capture the keys inside {}
    // Matches \cite, \parencite, \textcite, \footcite, etc.
    // Handles optional arguments like [page] before the brace
    const citationRegex = /\\(?:cite|parencite|textcite|footcite|citep|citet)[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}/g;
    
    let match;
    while ((match = citationRegex.exec(cleanContent)) !== null) {
      // match[1] contains the keys "key1, key2"
      const keys = match[1].split(',');
      keys.forEach(key => {
        const trimmed = key.trim();
        if (trimmed) {
          citations.add(trimmed);
        }
      });
    }
    
    return Array.from(citations);
  }

  /**
   * Removes specified citation keys from the LaTeX content.
   * If a citation command becomes empty (e.g. \cite{removed}), the command is removed.
   * If a citation command has multiple keys (e.g. \cite{kept, removed}), only the removed key is deleted.
   */
  public removeCitations(content: string, keysToRemove: string[]): string {
    if (keysToRemove.length === 0) return content;
    
    const keysSet = new Set(keysToRemove);
    const citationRegex = /(\\(?:cite|parencite|textcite|footcite|citep|citet)[a-zA-Z]*\*?(?:\[[^\]]*\])*)\{([^}]+)\}/g;

    return content.replace(citationRegex, (fullMatch, commandPrefix, keysBody) => {
      const currentKeys = keysBody.split(',').map((k: string) => k.trim());
      const keptKeys = currentKeys.filter((k: string) => !keysSet.has(k));

      if (keptKeys.length === 0) {
        // All keys removed, remove the entire citation command
        // We might want to leave a marker, but user asked to remove.
        return ''; 
      }

      // Reconstruct the citation with kept keys
      return `${commandPrefix}{${keptKeys.join(', ')}}`;
    });
  }

  /**
   * Parses LaTeX manuscript content into analyzed sentences.
   * @param content Raw LaTeX string
   */
  public parse(content: string): Result<AnalyzedSentence[], ParsingError> {
    try {
      if (!content || content.trim().length === 0) {
        return err(new ParsingError('Content is empty'));
      }

      // 0. Pre-clean non-text content (verbatim, iffalse, etc.)
      const cleanContent = this.removeNonTextContent(content);

      // 1. Placeholder map to protect citations from sentence splitting
      const placeholders: string[] = [];
      
      // Replace citations with placeholders __CITE_0__, __CITE_1__...
      let protectedText = cleanContent.replace(LatexParser.CITE_EXTRACT_REGEX, (match) => {
        placeholders.push(match);
        return `__CITE_${placeholders.length - 1}__`;
      });

      // 2. Clean other LaTeX artifacts
      protectedText = this.cleanLatex(protectedText);

      // 3. Split into sentences
      // Split on punctuation (.!?) that is followed by a space or end of string.
      const rawSentences = protectedText.match(/[^.!?]+[.!?]+(\s+|$)|[^.!?]+$/g) || [protectedText];

      const analyzedSentences = rawSentences
        .map(s => s.trim())
        .filter(s => s.length > 0)
        .map(sentenceWithPlaceholders => this.processSentence(sentenceWithPlaceholders, placeholders))
        .filter(s => s.text.length > 20); // Filter out tiny fragments

      return ok(analyzedSentences);

    } catch (error) {
      return err(new ParsingError('Unexpected error during LaTeX parsing', error));
    }
  }

  private cleanLatex(text: string): string {
    return text
      .replace(/%.*$/gm, '') // Remove comments
      .replace(/\\(sub)*section\*?\{([^}]+)\}/g, '$2. ') 
      .replace(/\\textbf\{([^}]+)\}/g, '$1')
      .replace(/\\textit\{([^}]+)\}/g, '$1')
      .replace(/\\emph\{([^}]+)\}/g, '$1')
      .replace(/\\label\{([^}]+)\}/g, '')
      .replace(/\s+/g, ' ');
  }

  private processSentence(sentenceWithPlaceholders: string, placeholders: string[]): AnalyzedSentence {
    const citations: string[] = [];
      
    const restoredSentence = sentenceWithPlaceholders.replace(/__CITE_(\d+)__/g, (match, index) => {
      const original = placeholders[parseInt(index, 10)];
      if (!original) return match;

      // Extract keys from the original citation string
      const keyMatch = /\{([^}]+)\}/.exec(original);
      if (keyMatch) {
        const keys = keyMatch[1].split(',').map(k => k.trim());
        citations.push(...keys);
      }
      
      return original;
    });

    // Rough check for numbers (digits or percentage signs)
    const hasNumbers = /\d+%?|\d+\.\d+/.test(restoredSentence);

    return {
      text: restoredSentence,
      citations,
      entities: [], // Will be filled by EntityExtractor later
      hasNumbers
    };
  }
}
