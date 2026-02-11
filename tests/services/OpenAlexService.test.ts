import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OpenAlexService } from '../../src/services/OpenAlexService';
import { ProcessedReference } from '../../src/types';

describe('OpenAlexService', () => {
  let service: OpenAlexService;

  beforeEach(() => {
    service = new OpenAlexService();
    global.fetch = vi.fn();
  });

  it('should fetch and map papers correctly', async () => {
    const mockResponse = {
      results: [
        {
          id: 'https://openalex.org/W123',
          title: 'Test Paper 1',
          publication_year: 2023,
          primary_location: { source: { display_name: 'Test Journal' } },
          authorships: [{ author: { display_name: 'Author 1' } }],
          abstract_inverted_index: { 'Abstract': [0], 'content': [1] }
        },
        {
          id: 'https://openalex.org/W456',
          title: 'Test Paper 2',
          publication_year: 2022,
          primary_location: { source: { display_name: 'Conference' } },
          authorships: [{ author: { display_name: 'Author 2' } }],
          abstract_inverted_index: { 'Another': [0], 'abstract': [1] }
        }
      ]
    };

    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    });

    const results = await service.searchPapers('test query', 2);

    expect(results).toHaveLength(2);
    expect(results[0].title).toBe('Test Paper 1');
    expect(results[0].abstract).toBe('Abstract content');
    expect(results[1].title).toBe('Test Paper 2');
    expect(results[1].abstract).toBe('Another abstract');
    
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('per_page=2')
    );
  });

  it('should return empty array on API error', async () => {
    (global.fetch as any).mockResolvedValue({
      ok: false,
      status: 500
    });

    const results = await service.searchPapers('test query');
    expect(results).toEqual([]);
  });
});
