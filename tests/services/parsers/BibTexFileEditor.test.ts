import { describe, it, expect } from 'vitest';
import { sortBibTexEntriesAlphabetically, upsertAndSortBibTexEntries } from '../../../src/services/parsers/BibTexFileEditor';

describe('BibTexFileEditor', () => {
  it('sorts entries alphabetically by citation key', () => {
    const input = `@article{bKey,
  title = {B}
}

@article{aKey,
  title = {A}
}
`;

    const output = sortBibTexEntriesAlphabetically(input);
    const aIndex = output.indexOf('@article{aKey');
    const bIndex = output.indexOf('@article{bKey');
    expect(aIndex).toBeGreaterThanOrEqual(0);
    expect(bIndex).toBeGreaterThanOrEqual(0);
    expect(aIndex).toBeLessThan(bIndex);
  });

  it('upserts new entries and keeps existing content', () => {
    const input = `\n% Prefix comment\n\n@article{Key1,\n  title = {One}\n}\n\n@article{Key3,\n  title = {Three}\n}\n\n% Suffix comment\n`;
    const newEntry = `@article{Key2,\n  title = {Two}\n}`;

    const output = upsertAndSortBibTexEntries(input, [newEntry]);
    expect(output).toContain('% Prefix comment');
    expect(output).toContain('% Suffix comment');
    expect(output).toContain('@article{Key2');

    const k1 = output.indexOf('@article{Key1');
    const k2 = output.indexOf('@article{Key2');
    const k3 = output.indexOf('@article{Key3');
    expect(k1).toBeLessThan(k2);
    expect(k2).toBeLessThan(k3);
  });

  it('does not duplicate an existing entry key', () => {
    const input = `@article{Dup,\n  title = {Original}\n}\n`;
    const dup = `@article{Dup,\n  title = {New}\n}`;
    const output = upsertAndSortBibTexEntries(input, [dup]);
    expect(output.match(/@article\{Dup,/g)?.length ?? 0).toBe(1);
    expect(output).toContain('Original');
    expect(output).not.toContain('New');
  });
});

