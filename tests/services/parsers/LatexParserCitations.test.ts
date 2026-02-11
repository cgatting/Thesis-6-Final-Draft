import { describe, it, expect } from 'vitest';
import { LatexParser } from '../../../src/services/parsers/LatexParser';

describe('LatexParser Citations', () => {
  const parser = new LatexParser();

  it('extracts standard citations', () => {
    const latex = `
      This is a claim \\cite{auth2023}.
      Another claim \\parencite{smith2022}.
    `;
    const keys = parser.extractCitations(latex);
    expect(keys).toContain('auth2023');
    expect(keys).toContain('smith2022');
    expect(keys.length).toBe(2);
  });

  it('extracts multiple keys from one command', () => {
    const latex = `See \\cite{key1, key2, key3} for details.`;
    const keys = parser.extractCitations(latex);
    expect(keys).toEqual(['key1', 'key2', 'key3']);
  });

  it('handles spaces in keys', () => {
    const latex = `See \\cite{ key1 ,  key2 }`;
    const keys = parser.extractCitations(latex);
    expect(keys).toEqual(['key1', 'key2']);
  });

  it('extracts from various citation commands', () => {
    const latex = `
      \\textcite{key1} says something.
      \\footcite{key2} is a footnote.
      \\citep{key3} and \\citet{key4}.
    `;
    const keys = parser.extractCitations(latex);
    expect(keys).toEqual(expect.arrayContaining(['key1', 'key2', 'key3', 'key4']));
  });

  it('handles optional arguments', () => {
    const latex = `\\cite[p. 23]{key1}`;
    const keys = parser.extractCitations(latex);
    expect(keys).toEqual(['key1']);
  });

  it('removes specific citation keys', () => {
    const latex = `\\cite{keep, remove}`;
    const result = parser.removeCitations(latex, ['remove']);
    expect(result).toBe('\\cite{keep}');
  });

  it('removes command if all keys removed', () => {
    const latex = `Text \\cite{remove} end.`;
    const result = parser.removeCitations(latex, ['remove']);
    expect(result).toBe('Text  end.');
  });

  it('removes multiple keys from multiple commands', () => {
    const latex = `
      \\cite{k1, r1}
      \\parencite{r2}
      \\textcite{k2, r1}
    `;
    const result = parser.removeCitations(latex, ['r1', 'r2']);
    
    // Whitespace might vary, so check for containment
    expect(result).toContain('\\cite{k1}');
    expect(result).not.toContain('r1');
    expect(result).not.toContain('r2');
    expect(result).toContain('\\textcite{k2}');
  });
});
