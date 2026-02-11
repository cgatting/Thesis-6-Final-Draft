import { describe, it, expect } from 'vitest';
import { LatexParser } from '../../../src/services/parsers/LatexParser';

describe('LatexParser', () => {
  const parser = new LatexParser();

  it('extracts title from \\title{...}', () => {
    const latex = `
      \\documentclass{article}
      \\title{The Effects of Climate Change}
      \\begin{document}
      Hello world.
    `;
    const title = parser.extractTitle(latex);
    expect(title).toBe('The Effects of Climate Change');
  });

  it('returns undefined if no title found', () => {
    const latex = `
      \\documentclass{article}
      \\begin{document}
      Hello world.
    `;
    const title = parser.extractTitle(latex);
    expect(title).toBeUndefined();
  });

  it('cleans latex formatting in title', () => {
    const latex = `\\title{\\textbf{Bold} and \\textit{Italic}}`;
    const title = parser.extractTitle(latex);
    expect(title).toBe('Bold and Italic');
  });

  it('truncates title at double backslash', () => {
    const latex = `\\title{Main Title \\\\ Subtitle}`;
    const title = parser.extractTitle(latex);
    expect(title).toBe('Main Title');
  });

  it('truncates title at newline command', () => {
    const latex = `\\title{Main Title \\newline Subtitle}`;
    const title = parser.extractTitle(latex);
    expect(title).toBe('Main Title');
  });
});
