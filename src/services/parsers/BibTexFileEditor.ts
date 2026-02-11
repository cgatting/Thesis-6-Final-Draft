export type BibTexEntryBlock = {
  key: string;
  raw: string;
  originalIndex: number;
};

export type BibTexSplitResult = {
  prefix: string;
  entries: BibTexEntryBlock[];
  suffix: string;
};

const SPECIAL_ENTRY_TYPES = new Set(['comment', 'preamble', 'string']);

export function splitBibTexFile(content: string): BibTexSplitResult {
  const prefixAndEntries = splitIntoEntryBlocks(content);
  if (prefixAndEntries.entries.length === 0) {
    return { prefix: content, entries: [], suffix: '' };
  }

  return {
    prefix: prefixAndEntries.prefix,
    entries: prefixAndEntries.entries,
    suffix: prefixAndEntries.suffix,
  };
}

export function sortBibTexEntriesAlphabetically(content: string): string {
  const { prefix, entries, suffix } = splitBibTexFile(content);
  if (entries.length === 0) return content;

  const sorted = [...entries].sort((a, b) => {
    const aIsSpecial = isSpecialEntry(a.raw);
    const bIsSpecial = isSpecialEntry(b.raw);

    if (aIsSpecial && bIsSpecial) return a.originalIndex - b.originalIndex;
    if (aIsSpecial) return -1;
    if (bIsSpecial) return 1;

    const aKey = a.key.toLowerCase();
    const bKey = b.key.toLowerCase();
    const byKey = aKey.localeCompare(bKey);
    if (byKey !== 0) return byKey;
    return a.originalIndex - b.originalIndex;
  });

  return prefix + sorted.map(e => e.raw).join('') + suffix;
}

export function upsertAndSortBibTexEntries(content: string, newEntryBlocks: string[]): string {
  const { prefix, entries, suffix } = splitBibTexFile(content);
  const existingKeys = new Set(entries.map(e => e.key));

  const additionalEntries: BibTexEntryBlock[] = [];
  for (const rawEntry of newEntryBlocks) {
    const parsed = parseSingleEntry(rawEntry);
    if (!parsed) continue;
    if (existingKeys.has(parsed.key)) continue;
    existingKeys.add(parsed.key);

    const normalizedRaw = ensureTrailingNewline(normalizeEntrySpacing(parsed.raw));
    additionalEntries.push({
      key: parsed.key,
      raw: (entries.length > 0 || additionalEntries.length > 0 || prefix.length > 0) ? ensureLeadingSeparator(normalizedRaw) : normalizedRaw,
      originalIndex: entries.length + additionalEntries.length,
    });
  }

  const merged = [...entries, ...additionalEntries].map((e, i) => ({ ...e, originalIndex: i }));
  const mergedText = prefix + merged.map(e => e.raw).join('') + suffix;
  return sortBibTexEntriesAlphabetically(mergedText);
}

export function extractBibFileNameFromTex(texContent: string): string | null {
  const addBibResource = texContent.match(/\\addbibresource\{([^}]+)\}/i);
  if (addBibResource?.[1]) return addBibResource[1].trim();

  const bibliography = texContent.match(/\\bibliography\{([^}]+)\}/i);
  if (bibliography?.[1]) {
    const name = bibliography[1].trim();
    return name.endsWith('.bib') ? name : `${name}.bib`;
  }

  return null;
}

export function serializeToBibTex(references: ProcessedReference[]): string {
  return references.map(ref => {
    return `@article{${ref.id},
  author = {${ref.authors.join(' and ')}},
  title = {${ref.title}},
  year = {${ref.year}},
  journal = {${ref.venue || ''}},
  abstract = {${ref.abstract}}
}`;
  }).join('\n\n');
}

function splitIntoEntryBlocks(content: string): BibTexSplitResult {
  const entries: BibTexEntryBlock[] = [];
  let prefix = '';
  let suffix = '';

  let i = 0;
  let firstEntryStart = -1;
  let prevEnd = 0;
  let entryIndex = 0;

  while (i < content.length) {
    const atIndex = content.indexOf('@', i);
    if (atIndex === -1) break;

    const parsed = parseEntryFromIndex(content, atIndex);
    if (!parsed) {
      i = atIndex + 1;
      continue;
    }

    if (firstEntryStart === -1) {
      firstEntryStart = atIndex;
      prefix = content.slice(0, firstEntryStart);
    }

    const trivia = content.slice(prevEnd, atIndex);
    const raw = trivia + content.slice(parsed.start, parsed.end);
    entries.push({ key: parsed.key, raw, originalIndex: entryIndex++ });

    prevEnd = parsed.end;
    i = parsed.end;
  }

  if (entries.length === 0) {
    return { prefix: content, entries: [], suffix: '' };
  }

  suffix = content.slice(prevEnd);
  return { prefix, entries, suffix };
}

function parseSingleEntry(rawEntry: string): { key: string; raw: string } | null {
  const trimmed = rawEntry.trimStart();
  const atIndex = trimmed.indexOf('@');
  if (atIndex === -1) return null;
  const parsed = parseEntryFromIndex(trimmed, atIndex);
  if (!parsed) return null;

  const raw = trimmed.slice(parsed.start, parsed.end).trim();
  return { key: parsed.key, raw };
}

function parseEntryFromIndex(content: string, start: number): { start: number; end: number; key: string } | null {
  if (content[start] !== '@') return null;

  let j = start + 1;
  while (j < content.length && /[A-Za-z]/.test(content[j])) j++;
  const type = content.slice(start + 1, j).trim().toLowerCase();
  if (!type) return null;

  while (j < content.length && /\s/.test(content[j])) j++;
  const opener = content[j];
  if (opener !== '{' && opener !== '(') return null;
  const closer = opener === '{' ? '}' : ')';
  const openerIndex = j;

  j++;
  while (j < content.length && /\s/.test(content[j])) j++;

  let key = '';
  if (!SPECIAL_ENTRY_TYPES.has(type)) {
    const keyStart = j;
    while (j < content.length && content[j] !== ',' && content[j] !== closer) j++;
    key = content.slice(keyStart, j).trim();
    if (!key) return null;
  } else {
    key = `@${type}:${start}`;
  }

  const end = scanBalancedBlockEnd(content, openerIndex, opener, closer);
  if (end === null) return null;
  return { start, end, key };
}

function scanBalancedBlockEnd(content: string, openerIndex: number, opener: string, closer: string): number | null {
  let depth = 0;
  let inQuote = false;
  for (let i = openerIndex; i < content.length; i++) {
    const ch = content[i];
    const prev = i > 0 ? content[i - 1] : '';

    if (ch === '"' && prev !== '\\') inQuote = !inQuote;
    if (inQuote) continue;

    if (ch === opener) depth++;
    else if (ch === closer) {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  return null;
}

function ensureTrailingNewline(text: string): string {
  return text.endsWith('\n') ? text : `${text}\n`;
}

function ensureLeadingSeparator(entry: string): string {
  if (entry.startsWith('\n')) return entry;
  return `\n\n${entry}`;
}

function normalizeEntrySpacing(entry: string): string {
  const trimmed = entry.trim();
  return trimmed;
}

function isSpecialEntry(raw: string): boolean {
  const m = raw.match(/@\s*([A-Za-z]+)/);
  const t = m?.[1]?.toLowerCase();
  return t ? SPECIAL_ENTRY_TYPES.has(t) : false;
}

