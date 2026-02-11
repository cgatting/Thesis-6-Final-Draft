import { describe, it, expect } from 'vitest';
import { BibTexParser } from '../../../src/services/parsers/BibTexParser';
import { unwrap } from '../../../src/utils/result';

describe('BibTexParser', () => {
  it('should parse standard bibtex entries', () => {
    const bib = `
      @article{doe2023,
        author = {John Doe and Jane Smith},
        title = {Deep Learning},
        year = {2023},
        abstract = {This is a great paper.}
      }
    `;
    const parser = new BibTexParser();
    const result = parser.parse(bib);
    
    expect(result.ok).toBe(true);
    const refs = unwrap(result);
    expect(refs.length).toBe(1);
    expect(refs[0].id).toBe('doe2023');
    expect(refs[0].authors).toEqual(['John Doe', 'Jane Smith']);
  });
  
  it('should fallback to simple line parsing if bibtex fails', () => {
      const text = `
      [1] Doe (2023). A Title.
      [2] Smith (2022). Another Title.
      `;
      const parser = new BibTexParser();
      const result = parser.parse(text);
      expect(result.ok).toBe(true);
      const refs = unwrap(result);
      expect(refs.length).toBe(2);
  });
});
