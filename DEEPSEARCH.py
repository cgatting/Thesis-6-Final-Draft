import argparse
import logging
import os
import time
import re
import requests
import threading
import multiprocessing
import concurrent.futures
import math
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional
import asyncio
import json
import tempfile
import unittest
import aiohttp
from unittest.mock import patch
# Lazy load these heavy imports
# import nltk
# import torch
# import yake
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from sklearn.metrics.pairwise import cosine_similarity
import customtkinter as ctk
from tkinter import filedialog, messagebox, colorchooser
import tkinter.ttk as ttk
import webbrowser
from urllib.parse import quote
import sys

# Placeholders for lazy loading
nltk = None
torch = None
yake = None
SentenceTransformer = None
pipeline = None
cosine_similarity = None

def _lazy_load_ml_deps():
    global nltk, torch, yake, SentenceTransformer, pipeline, cosine_similarity
    if torch is not None:
        return

    logging.info("Loading ML dependencies...")
    import nltk as _nltk
    import torch as _torch
    import yake as _yake
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    from transformers import pipeline as _pipeline
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    
    nltk = _nltk
    torch = _torch
    yake = _yake
    SentenceTransformer = _SentenceTransformer
    pipeline = _pipeline
    cosine_similarity = _cosine_similarity
    
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logging.error(f"Error downloading NLTK data: {e}")

    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    logging.info("ML dependencies loaded.")

# ---------------- Logging Setup ----------------
LOG_FILE = 'deep_research_tool_improved.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# NLTK download and torch setup moved to _lazy_load_ml_deps

# Add memory management constants
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 100 * 1024  # 100KB for processing chunks
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.deep_research_cache')

# Add citation format templates
CITATION_FORMATS = {
    'APA': "{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}. DOI: https://doi.org/{doi}",
    'MLA': "{authors}. \"{title}.\" {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}. DOI: https://doi.org/{doi}",
    'Chicago': "{authors}. {year}. \"{title}.\" {journal} {volume}, no. {issue}: {pages}. DOI: https://doi.org/{doi}",
    'Harvard': "{authors} ({year}) '{title}', {journal}, {volume}({issue}), pp. {pages}. Available at: https://doi.org/{doi}"
}

# ---------------- NLP Processor ----------------
class NLPProcessor:
    def __init__(self, settings: dict):
        _lazy_load_ml_deps()
        self.settings = settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.sentence_model = SentenceTransformer(settings['model_settings']['sentence_model'])
            if torch.cuda.is_available():
                self.sentence_model = self.sentence_model.to(self.device)
            self.keyword_extractor = yake.KeywordExtractor(**settings['keyword_settings'])
            device_id = -1
            self.summarizer = pipeline(
                "summarization",
                model=settings['model_settings']['summarizer_model'],
                device=device_id
            )
        except Exception as e:
            logging.error(f"Error initializing NLP models: {e}")
            raise

    @lru_cache(maxsize=1000)
    def refine_query(self, sentence: str) -> str:
        try:
            # Extract keywords only, avoiding verbose summaries for search queries
            keywords = [kw[0] for kw in self.keyword_extractor.extract_keywords(sentence)]
            # Take top 5 keywords to keep query focused
            refined = " ".join(keywords[:5]).strip()
            logging.info(f"Refined query: {refined}")
            return refined
        except Exception as e:
            logging.error(f"Error refining query: {e}")
            return sentence

    @lru_cache(maxsize=1000)
    def calculate_similarity(self, text1: str, text2: str) -> float:
        try:
            with torch.no_grad():
                emb1 = self.sentence_model.encode(text1, convert_to_tensor=True)
                emb2 = self.sentence_model.encode(text2, convert_to_tensor=True)
                if self.device.type == 'cuda':
                    emb1 = emb1.cpu()
                    emb2 = emb2.cpu()
                sim = float(cosine_similarity(
                    emb1.numpy().reshape(1, -1),
                    emb2.numpy().reshape(1, -1)
                )[0][0])
                return sim
        except Exception as e:
            logging.error(f"Error calculating similarity: {e}")
            return 0.0

    def encode_text(self, text: str):
        """Encode a single text to a numpy array."""
        try:
            with torch.no_grad():
                emb = self.sentence_model.encode(text, convert_to_tensor=True)
                if self.device.type == 'cuda':
                    emb = emb.cpu()
                return emb.numpy().reshape(1, -1)
        except Exception as e:
            logging.error(f"Error encoding text: {e}")
            return None

    def calculate_similarity_with_embedding(self, emb1, text2: str) -> float:
        """Calculate similarity using a pre-computed embedding for the first text."""
        try:
            emb2 = self.encode_text(text2)
            if emb2 is None: return 0.0
            return float(cosine_similarity(emb1, emb2)[0][0])
        except Exception:
            return 0.0

# ---------------- Async Document Processor ----------------
class AsyncDocumentProcessor:
    """Handles asynchronous document processing with chunking support."""
    def __init__(self, chunk_size=CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.processed_chunks = []
        
    async def process_document(self, text: str, processor_func) -> str:
        chunks = self._split_into_chunks(text)
        tasks = [self._process_chunk(chunk, processor_func) for chunk in chunks]
        processed_chunks = await asyncio.gather(*tasks)
        return self._merge_chunks(processed_chunks)
    
    def _split_into_chunks(self, text: str) -> List[str]:
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
    
    async def _process_chunk(self, chunk: str, processor_func) -> str:
        return await processor_func(chunk)
    
    def _merge_chunks(self, chunks: List[str]) -> str:
        return ''.join(chunks)

# ---------------- Document Cache ----------------
class DocumentCache:
    """Handles caching of processed documents and search results."""
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cached_result(self, key: str) -> dict:
        cache_file = os.path.join(self.cache_dir, f"{hash(key)}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Cache read error: {e}")
        return None
    
    def cache_result(self, key: str, data: dict):
        cache_file = os.path.join(self.cache_dir, f"{hash(key)}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logging.warning(f"Cache write error: {e}")

# ---------------- Source Ranking Engine ----------------
class SourceRankingEngine:
    def __init__(self, settings: dict, nlp_processor: NLPProcessor, search_engine: Any, cache: Optional[DocumentCache] = None):
        self.settings = settings
        self.nlp = nlp_processor
        self.search = search_engine
        self.cache = cache or DocumentCache()

    def extract_key_terms(self, text: str, top_k: int = 15) -> List[str]:
        try:
            kws = self.nlp.keyword_extractor.extract_keywords(text)
            terms = [kw for kw, _ in sorted(kws, key=lambda x: x[1])][:top_k]
            return terms
        except Exception:
            words = re.findall(r"[A-Za-z]{4,}", text.lower())
            freq: Dict[str, int] = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_k]]

    def build_query(self, text: str, use_refine: bool) -> str:
        if use_refine:
            return self.nlp.refine_query(text)
        terms = self.extract_key_terms(text)
        return " ".join(terms)

    def score_result(self, doc_text: str, item: Dict[str, Any], terms: List[str]) -> float:
        title = (item.get('title') or [''])[0]
        abstract = item.get('abstract', '')
        combined = f"{title} {abstract}".strip()
        sim = self.nlp.calculate_similarity(doc_text, combined) if combined else 0.0
        
        # Return 0 if similarity is too low to be relevant
        if sim < 0.2:
            return 0.0

        lower_combined = combined.lower()
        overlap = sum(1 for t in terms if t.lower() in lower_combined)
        overlap_norm = overlap / max(1, len(terms))
        
        # Quality metrics
        citations = item.get('is-referenced-by-count', 0)
        citation_score = min(1.0, math.log10(citations + 1) / 4.0)  # Normalize log(citations)
        
        current_year = datetime.now().year
        try:
            year = item.get('published-print', {}).get('date-parts', [[None]])[0][0] or \
                   item.get('published-online', {}).get('date-parts', [[None]])[0][0]
            age = current_year - int(year) if year else 100
            if age <= 3: recency_score = 1.0
            elif age <= 5: recency_score = 0.8
            elif age <= 10: recency_score = 0.5
            else: recency_score = 0.2
        except:
            recency_score = 0.2

        # Weighted score
        weights = self.settings.get('ranking_settings', {})
        w_sim = weights.get('weight_similarity', 0.65)
        w_ovl = weights.get('weight_overlap', 0.1)
        w_cit = weights.get('weight_citations', 0.15)
        w_rec = weights.get('weight_recency', 0.1)
        
        return w_sim * sim + w_ovl * overlap_norm + w_cit * citation_score + w_rec * recency_score

    def summarize_item(self, item: Dict[str, Any]) -> str:
        try:
            text = " ".join([s for s in [item.get('title', [''])[0], item.get('abstract', '')] if s])
            if not text:
                return ""
            out = self.nlp.summarizer(
                text,
                max_length=min(48, self.settings['model_settings']['max_length']),
                min_length=min(12, self.settings['model_settings']['min_length']),
                do_sample=False
            )[0]['summary_text']
            return out
        except Exception:
            return (item.get('title', [''])[0] or '')[:120]

    def apply_filters(self, items: List[Dict]) -> List[Dict]:
        filtered = []
        search_settings = self.settings.get('search_settings', {})
        min_year = search_settings.get('min_year')
        min_citations = search_settings.get('min_citations', 0)
        
        for item in items:
            # Citation Count Check
            if item.get('is-referenced-by-count', 0) < min_citations:
                continue
                
            # Year Check
            if min_year:
                try:
                    # Try print date first, then online date
                    date_parts = item.get('published-print', {}).get('date-parts') or \
                                 item.get('published-online', {}).get('date-parts')
                    if date_parts:
                        pub_year = date_parts[0][0]
                        if pub_year and int(pub_year) < int(min_year):
                            continue
                except Exception:
                    # If we can't determine year, decide whether to keep or drop.
                    # Usually drop if strict filtering is on.
                    pass
            
            filtered.append(item)
        return filtered

    async def rank_top_sources(self, doc_text: str, rows: int, threshold: float, use_refine: bool, progress_cb: Optional[Any] = None) -> List[Dict[str, Any]]:
        # Include snowballing setting in cache key
        snowball = self.settings.get('search_settings', {}).get('enable_snowballing', False)
        key = json.dumps({"text_hash": hash(doc_text), "rows": rows, "thr": threshold, "ref": use_refine, "snow": snowball}, sort_keys=True)
        cached = self.cache.get_cached_result(key)
        if cached:
            return cached.get('results', [])

        if progress_cb:
            progress_cb(0.05, "Extracting key terms...")
        terms = self.extract_key_terms(doc_text)
        query = self.build_query(doc_text, use_refine)

        if progress_cb:
            progress_cb(0.2, "Searching sources...")
        items = await self.search.search_papers(query, limit=rows)
        
        # Apply Filters
        items = self.apply_filters(items)

        if progress_cb:
            progress_cb(0.45, f"Scoring {len(items)} results...")
        
        scored: List[Dict[str, Any]] = []
        existing_dois = set()

        def process_and_score(candidate_items):
            for it in candidate_items:
                doi = it.get('DOI')
                if not doi or not it.get('container-title') or doi in existing_dois:
                    continue
                score = self.score_result(doc_text, it, terms)
                if score >= threshold:
                    scored.append({"item": it, "score": float(round(score, 4))})
                    existing_dois.add(doi)

        process_and_score(items)

        # Snowballing Logic
        if snowball and len(scored) > 0:
            if progress_cb:
                progress_cb(0.6, "Performing snowball sampling (references)...")
            
            # Take top 3 high scoring items as seeds
            top_seeds = sorted(scored, key=lambda x: -x['score'])[:3]
            new_items = []
            
            for seed in top_seeds:
                seed_doi = seed['item'].get('DOI')
                # Fetch references
                refs = await self.search.get_references(seed_doi)
                # Limit to 5 references per seed to check
                for ref in refs[:5]:
                    ref_doi = ref.get('DOI')
                    if ref_doi and ref_doi not in existing_dois:
                        details = await self.search.fetch_paper_details(ref_doi)
                        if details:
                            new_items.append(details)
            
            if new_items:
                new_items = self.apply_filters(new_items)
                process_and_score(new_items)

        if progress_cb:
            progress_cb(0.8, "Selecting top 10 and summarizing...")
            
        top = sorted(scored, key=lambda x: -x['score'])[:10]
        results: List[Dict[str, Any]] = []
        for t in top:
            it = t['item']
            authors = []
            for a in it.get('author', []) or []:
                fam = a.get('family', '')
                giv = a.get('given', '')
                if fam or giv:
                    authors.append(f"{fam}, {giv}".strip(', '))
            year = None
            try:
                year = it.get('published-print', {}).get('date-parts', [[None]])[0][0] or it.get('published-online', {}).get('date-parts', [[None]])[0][0]
            except Exception:
                year = None
            link = it.get('URL') or (f"https://doi.org/{it.get('DOI')}" if it.get('DOI') else "")
            summary = self.summarize_item(it)
            results.append({
                "title": (it.get('title') or [''])[0],
                "authors": "; ".join(authors),
                "year": str(year or ''),
                "journal": (it.get('container-title') or [''])[0],
                "doi": it.get('DOI', ''),
                "score": t['score'],
                "summary": summary,
                "link": link
            })

        self.cache.cache_result(key, {"results": results})
        if progress_cb:
            progress_cb(0.95, "Done")
        return results

# ---------------- Research Search Engine ----------------
class ResearchSearchEngine:
    def __init__(self):
        self.headers = {'User-Agent': 'DeepResearchTool/1.0 (mailto:your.email@example.com)'}
        self.max_retries = 3  # Reduced retries for speed
        self.timeout = 15     # Reduced timeout for speed
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Reduced interval, use async sleep
        self.semaphore = asyncio.Semaphore(10) # Limit concurrent requests

    async def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits by waiting between requests"""
        async with self.semaphore:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last_request)
            self.last_request_time = time.time()

    async def search_papers(self, query: str, limit: int = 500) -> List[Dict]:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            for attempt in range(self.max_retries):
                try:
                    await self._wait_for_rate_limit()
                    async with session.get(
                        'https://api.crossref.org/works',
                        params={'query': query, 'rows': limit},
                        timeout=self.timeout
                    ) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get('Retry-After', 5))
                            logging.warning(f"Rate limited by CrossRef. Waiting {retry_after} seconds.")
                            await asyncio.sleep(retry_after)
                            continue
                            
                        response.raise_for_status()
                        data = await response.json()
                        items = data.get('message', {}).get('items', [])
                        logging.info(f"Found {len(items)} items for query: {query}")
                        return items
                    
                except Exception as e:
                    wait_time = min(10, 2 ** attempt)
                    logging.error(f"CrossRef search error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        return []
                    logging.info(f"Waiting {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)
        return []

    async def get_references(self, doi: str) -> List[Dict]:
        """Fetch references for a given DOI (Snowballing step 1)."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                await self._wait_for_rate_limit()
                url = f"https://api.crossref.org/works/{doi}"
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('message', {}).get('reference', [])
            except Exception as e:
                logging.warning(f"Failed to fetch references for {doi}: {e}")
        return []

    async def fetch_paper_details(self, doi: str) -> Optional[Dict]:
        """Fetch full metadata for a specific DOI (Snowballing step 2)."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            try:
                await self._wait_for_rate_limit()
                url = f"https://api.crossref.org/works/{doi}"
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('message')
            except Exception:
                pass
        return None

# ---------------- Document Refiner ----------------
class DocumentRefiner:
    def __init__(self, settings: dict, view):
        self.settings = settings
        self.view = view
        self.nlp_processor = NLPProcessor(settings)
        self.search_engine = ResearchSearchEngine()
        self.bibliography = {}
        self.bib_lock = threading.Lock()
        self.doc_cache = DocumentCache()
        self.async_processor = AsyncDocumentProcessor()
        
    async def process_sentence_async(self, sentence: str, idx: int) -> str:
        """Asynchronous version of process_sentence with improved error handling and caching."""
        if not self.sentence_needs_citation(sentence):
            return sentence
            
        cache_key = f"sentence_{hash(sentence)}"
        cached_result = self.doc_cache.get_cached_result(cache_key)
        if cached_result:
            return cached_result['processed_sentence']
            
        try:
            query = await self._refine_query_async(sentence)
            search_results = await self._search_papers_async(query)
            
            best_result = await self._find_best_match_async(sentence, search_results)
            if best_result:
                citation = self._create_citation(best_result)
                processed_sentence = f"{sentence} {citation}"
                self.doc_cache.cache_result(cache_key, {
                    'processed_sentence': processed_sentence,
                    'citation_data': best_result
                })
                return processed_sentence
        except Exception as e:
            logging.error(f"Error processing sentence: {e}")
            self.view.show_error(f"Error processing sentence: {str(e)}")
        return sentence

    async def refine_document(self, doc_text: str) -> str:
        """Improved document refinement with parallel async processing."""
        if len(doc_text) > MAX_DOCUMENT_SIZE:
            raise ValueError(f"Document size exceeds maximum limit of {MAX_DOCUMENT_SIZE/1024/1024}MB")
            
        sentences = nltk.sent_tokenize(doc_text)
        total = len(sentences)
        processed_sentences = [None] * total  # Pre-allocate list
        
        # Limit concurrency to avoid overwhelming the system
        sem = asyncio.Semaphore(10)
        completed_count = 0
        
        async def process_with_sem(sent, idx):
            nonlocal completed_count
            async with sem:
                res = await self.process_sentence_async(sent, idx)
                completed_count += 1
                self.view.update_progress(completed_count / total, f"Refining manuscript... ({completed_count}/{total} sentences processed)")
                return idx, res

        tasks = [process_with_sem(s, i) for i, s in enumerate(sentences)]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x[0])
        processed_sentences = [r[1] for r in results]
            
        return "\n\n".join(processed_sentences)

    def sentence_needs_citation(self, sentence: str) -> bool:
        """Heuristic to determine if a sentence needs a citation."""
        if '\\cite' in sentence:
            return False
        words = sentence.split()
        if len(words) < 5:
            return False
        # Trigger words that often indicate a claim or factual statement.
        triggers = [
            'find', 'finds', 'found', 'suggest', 'suggests', 'reported', 'reports',
            'demonstrate', 'demonstrates', 'evidence', 'conclude', 'concludes',
            'indicate', 'indicates', 'study', 'studies', 'analysis', 'analyses',
            'data', 'research', 'observed', 'observes', 'observing'
        ]
        sentence_lower = sentence.lower()
        if any(trigger in sentence_lower for trigger in triggers):
            return True
        # Also, if the sentence contains numbers (which may indicate data or statistics)
        if any(char.isdigit() for char in sentence):
            return True
        return False

    def create_citation_id(self, result: Dict) -> str:
        try:
            author_family = result.get('author', [{}])[0].get('family', 'Unknown')
            year = result.get('published-print', {}).get('date-parts', [['Unknown']])[0][0]
        except Exception:
            author_family, year = "Unknown", "Unknown"
        citation_id = f"{author_family}{year}"
        return citation_id

    async def _refine_query_async(self, sentence: str) -> str:
        return self.nlp_processor.refine_query(sentence)

    async def _search_papers_async(self, query: str) -> List[Dict]:
        return self.search_engine.search_papers(query, limit=self.settings.get('search_settings', {}).get('max_results', 500))

    async def _find_best_match_async(self, sentence: str, search_results: List[Dict]) -> Dict:
        best_result = None
        best_score = 0.0
        current_year = datetime.now().year

        # Pre-compute sentence embedding
        sentence_emb = self.nlp_processor.encode_text(sentence)
        if sentence_emb is None:
            return None

        for result in search_results:
            # Additional quality checks: only consider results with DOI and container-title.
            if not result.get('DOI') or not result.get('container-title', []):
                continue
            title = result.get('title', [''])[0]
            abstract = result.get('abstract', '')
            combined_text = f"{title} {abstract}"
            similarity = self.nlp_processor.calculate_similarity_with_embedding(sentence_emb, combined_text)
            
            # Use lowered threshold from previous fix
            threshold = self.settings.get('similarity_threshold', 0.5)
            if similarity < threshold:
                continue

            # Calculate Quality Score
            citations = result.get('is-referenced-by-count', 0)
            citation_score = min(1.0, math.log10(citations + 1) / 4.0)
            
            recency_score = 0.2
            try:
                year = result.get('published-print', {}).get('date-parts', [[None]])[0][0] or \
                       result.get('published-online', {}).get('date-parts', [[None]])[0][0]
                if year:
                    age = current_year - int(year)
                    if age <= 3: recency_score = 1.0
                    elif age <= 5: recency_score = 0.8
                    elif age <= 10: recency_score = 0.5
            except:
                pass

            # Boost similarity with quality metrics
            # Formula: Sim * (1 + QualityBoosts)
            final_score = similarity * (1.0 + 0.2 * citation_score + 0.1 * recency_score)

            logging.info(f"Candidate: '{title[:30]}...' Sim={similarity:.3f}, Cit={citations}, Score={final_score:.3f}")
            
            if final_score > best_score:
                best_score = final_score
                best_result = result
        return best_result

    def _create_citation(self, result: Dict) -> str:
        citation_id = self.create_citation_id(result)
        with self.bib_lock:
            if citation_id not in self.bibliography:
                authors = " and ".join([
                    f"{author.get('family', '')}, {author.get('given', '')}"
                    for author in result.get('author', [])
                ])
                self.bibliography[citation_id] = {
                    'ID': citation_id,
                    'title': result.get('title', [''])[0],
                    'authors': authors,
                    'year': str(result.get('published-print', {}).get('date-parts', [['Unknown']])[0][0]),
                    'doi': result.get('DOI', 'N/A'),
                    'journal': result.get('container-title', [''])[0],
                    'volume': result.get('volume', ''),
                    'issue': result.get('issue', ''),
                    'pages': result.get('page', ''),
                    'link': result.get('URL') or (f"https://doi.org/{result.get('DOI')}" if result.get('DOI') else "")
                }
        
        # Check citation format from settings
        # Always use LaTeX citation format as requested
        return f"\\citep{{{citation_id}}}"

    def generate_bibliography_text(self) -> str:
        bib_entries = []
        citation_format = self.settings.get('ui_settings', {}).get('citation_format', 'APA')
        
        with self.bib_lock:
            # Sort bibliography by author
            sorted_bib = sorted(self.bibliography.values(), key=lambda x: x['authors'])
            
            for entry in sorted_bib:
                fmt_str = CITATION_FORMATS.get(citation_format, CITATION_FORMATS['APA'])
                bib_text = fmt_str.format(
                    authors=entry['authors'],
                    year=entry['year'],
                    title=entry['title'],
                    journal=entry.get('container-title', [''])[0],
                    volume=entry.get('volume', ''),
                    issue=entry.get('issue', ''),
                    pages=entry.get('page', ''),
                    doi=entry['doi']
                )
                bib_entries.append(bib_text)
        
        # Add link to references.bib if there are entries
        if bib_entries:
            bib_entries.append("\nFull references available in: references.bib")
            
        return "\n\n".join(bib_entries)

    def generate_bibtex_content(self) -> str:
        entries = []
        with self.bib_lock:
            # Sort bibliography by author
            sorted_bib = sorted(self.bibliography.values(), key=lambda x: x['authors'])
            
            for entry in sorted_bib:
                key = entry['ID']
                # Use global escape_bibtex if available, otherwise simple fallback
                try:
                    title = escape_bibtex(entry.get('title', ''))
                    authors = escape_bibtex(entry.get('authors', ''))
                    journal = escape_bibtex(entry.get('journal', ''))
                    year = escape_bibtex(entry.get('year', ''))
                    volume = escape_bibtex(entry.get('volume', ''))
                    number = escape_bibtex(entry.get('issue', ''))
                    pages = escape_bibtex(entry.get('pages', ''))
                    url = escape_bibtex(entry.get('link', ''))
                except NameError:
                    # Fallback if escape_bibtex is not defined yet
                    title = entry.get('title', '')
                    authors = entry.get('authors', '')
                    journal = entry.get('journal', '')
                    year = entry.get('year', '')
                    volume = entry.get('volume', '')
                    number = entry.get('issue', '')
                    pages = entry.get('pages', '')
                    url = entry.get('link', '')

                doi = str(entry.get('doi', '')).strip()
                
                lines = [f"@article{{{key},"]
                lines.append(f"  title = {{{{{title}}}}},")
                if authors:
                    lines.append(f"  author = {{{authors}}},")
                if journal:
                    lines.append(f"  journal = {{{journal}}},")
                if year:
                    lines.append(f"  year = {{{year}}},")
                if volume:
                    lines.append(f"  volume = {{{volume}}},")
                if number:
                    lines.append(f"  number = {{{number}}},")
                if pages:
                    lines.append(f"  pages = {{{pages}}},")
                if doi and doi != 'N/A':
                    lines.append(f"  doi = {{{doi}}},")
                if url:
                    lines.append(f"  url = {{{url}}},")
                
                # remove trailing comma
                if lines[-1].endswith(','):
                    lines[-1] = lines[-1][:-1]
                lines.append('}')
                entries.append('\n'.join(lines))
                
        return '\n\n'.join(entries)

    def generate_ris_content(self) -> str:
        lines = []
        with self.bib_lock:
            sorted_bib = sorted(self.bibliography.values(), key=lambda x: x['authors'])
            for entry in sorted_bib:
                lines.append("TY  - JOUR")
                lines.append(f"TI  - {entry.get('title', '')}")
                # Authors need splitting? "Smith, John and Doe, Jane"
                authors = entry.get('authors', '').split(' and ')
                for au in authors:
                    lines.append(f"AU  - {au.strip()}")
                lines.append(f"PY  - {entry.get('year', '')}")
                lines.append(f"JO  - {entry.get('journal', '')}")
                lines.append(f"VL  - {entry.get('volume', '')}")
                lines.append(f"IS  - {entry.get('issue', '')}")
                lines.append(f"SP  - {entry.get('pages', '')}")
                doi = entry.get('doi', '')
                if doi and doi != 'N/A':
                    lines.append(f"DO  - {doi}")
                lines.append("ER  - ")
                lines.append("")
        return "\n".join(lines)

    def generate_research_report(self) -> str:
        report = []
        report.append("RESEARCH METHODOLOGY REPORT")
        report.append("===========================")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")
        report.append("1. Search Configuration")
        report.append("----------------------")
        search_settings = self.settings.get('search_settings', {})
        report.append(f"Similarity Threshold: {self.settings.get('similarity_threshold')}")
        report.append(f"Min Citations: {search_settings.get('min_citations', 0)}")
        report.append(f"Min Year: {search_settings.get('min_year', 'None')}")
        report.append(f"Snowballing Enabled: {search_settings.get('enable_snowballing', False)}")
        report.append("")
        report.append("2. Citation Statistics")
        report.append("---------------------")
        with self.bib_lock:
            count = len(self.bibliography)
            report.append(f"Total Citations Found: {count}")
        return "\n".join(report)

# ---------------- Deep Research Tool App (GUI) ----------------
class DeepResearchToolApp:
    def __init__(self, settings: dict):
        self.settings = settings
        self.document_text = ""   # Original document text
        self.refined_text = ""    # Updated document text after processing
        self.doc_refiner = None   # Will be initialized after UI setup
        self.async_loop = None    # Will store the asyncio event loop
        self.source_ranking = None
        self.search_engine = None
        
        # Initialize UI
        self.setup_ui()
        
        # Initialize processing components
        try:
            self.doc_refiner = DocumentRefiner(settings, self)
            self.search_engine = self.doc_refiner.search_engine
            self.source_ranking = SourceRankingEngine(settings, self.doc_refiner.nlp_processor, self.search_engine, self.doc_refiner.doc_cache)
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
            threading.Thread(target=self._run_async_loop, daemon=True).start()
        except Exception as e:
            self.show_error(f"Failed to initialize document processor: {e}")
            raise
        
        # Setup additional UI components that depend on doc_refiner
        self.setup_keyboard_shortcuts()
        self.load_theme_preference()
        
    def _run_async_loop(self):
        """Run the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.async_loop)
        self.async_loop.run_forever()
        
    def setup_ui(self):
        """Initialize the UI components."""
        # Set initial theme from settings
        theme = self.settings.get('ui_settings', {}).get('theme', 'dark')
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Deep Research Document Refiner")
        window_size = self.settings.get('ui_settings', {}).get('window_size', '1000x750')
        self.root.geometry(window_size)
        
        # Create UI frames
        self.create_control_frame()
        self.create_progress_frame()
        self.create_notebook_frame()
        self.create_console_frame()
        
        # Make text boxes read-only initially
        self.orig_textbox.configure(state="disabled")
        self.refined_textbox.configure(state="disabled")
        self.bib_textbox.configure(state="disabled")
        
        # Add tooltips after all widgets are created
        self.add_tooltips()
        
        # Configure window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Log initial status
        self.log_to_console("Application initialized successfully")
        
    def on_closing(self):
        """Handle application shutdown."""
        try:
            if self.async_loop and self.async_loop.is_running():
                self.async_loop.stop()
            if hasattr(self, 'root'):
                self.root.destroy()
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)

    def add_tooltips(self):
        """Add tooltips to UI elements."""
        self.tooltips = {}
        tooltip_texts = {
            self.upload_button: "Upload Document (Ctrl+O)",
            self.refine_button: "Start Refinement (Ctrl+R)",
            self.save_button: "Save Document (Ctrl+S)",
            self.bib_button: "Save Bibliography (Ctrl+B)",
            self.theme_button: "Toggle Dark/Light Theme",
            self.citation_format: "Select Citation Format"
        }
        
        for widget, text in tooltip_texts.items():
            try:
                tooltip = ctk.CTkToolTip(widget, message=text)
                self.tooltips[widget] = tooltip
            except Exception as e:
                logging.warning(f"Failed to create tooltip: {e}")

    def upload_document(self, event=None):
        """Handle document upload."""
        try:
            filename = filedialog.askopenfilename(
                title="Select Document",
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("LaTeX Files", "*.tex"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.document_text = f.read()
                
                # Update original text display
                self.orig_textbox.configure(state="normal")
                self.orig_textbox.delete("1.0", "end")
                self.orig_textbox.insert("end", self.document_text)
                self.orig_textbox.configure(state="disabled")
                
                # Enable refinement button
                self.refine_button.configure(state="normal")
                
                # Reset other UI elements
                self.refined_textbox.configure(state="normal")
                self.refined_textbox.delete("1.0", "end")
                self.refined_textbox.configure(state="disabled")
                
                self.bib_textbox.configure(state="normal")
                self.bib_textbox.delete("1.0", "end")
                self.bib_textbox.configure(state="disabled")
                
                self.save_button.configure(state="disabled")
                self.bib_button.configure(state="disabled")
                
                self.progress_bar.set(0)
                self.status_label.configure(text="Ready to process")
                
                self.log_to_console(f"Loaded document: {filename}")
        except Exception as e:
            self.show_error(f"Error loading document: {e}")
            
    def save_refined_document(self, event=None):
        """Save the refined document."""
        if not self.refined_text:
            self.show_error("No refined document to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Refined Document",
                defaultextension=".txt",
                filetypes=[
                    ("Text Files", "*.txt"),
                    ("LaTeX Files", "*.tex"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.refined_text)
                self.log_to_console(f"Saved refined document to: {filename}")
        except Exception as e:
            self.show_error(f"Error saving document: {e}")

    def save_bibliography(self, event=None):
        """Save the bibliography."""
        if not self.doc_refiner or not self.doc_refiner.bibliography:
            self.show_error("No bibliography to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Bibliography",
                defaultextension=".bib",
                filetypes=[("BibTeX Files", "*.bib"), ("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if filename:
                if filename.lower().endswith('.bib'):
                    content = self.doc_refiner.generate_bibtex_content()
                else:
                    content = self.doc_refiner.generate_bibliography_text()
                    
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.log_to_console(f"Saved bibliography to: {filename}")
        except Exception as e:
            self.show_error(f"Error saving bibliography: {e}")

    def update_progress(self, fraction: float, status: str):
        """Update the progress bar and status label."""
        try:
            self.progress_bar.set(fraction)
            self.status_label.configure(text=status)
            self.root.update_idletasks()
        except Exception as e:
            logging.error(f"Error updating progress: {e}")

    def log_to_console(self, message: str):
        """Add a message to the console with timestamp."""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.console.configure(state="normal")
            self.console.insert("end", f"[{timestamp}] {message}\n")
            self.console.see("end")
            self.console.configure(state="disabled")
        except Exception as e:
            logging.error(f"Error logging to console: {e}")

    def show_error(self, message: str):
        """Display error message in console and dialog."""
        self.log_to_console(f"ERROR: {message}")
        threading.Thread(target=lambda: messagebox.showerror("Error", message)).start()

    def toggle_theme(self):
        """Toggle between light and dark theme."""
        try:
            current_theme = ctk.get_appearance_mode()
            new_theme = "Light" if current_theme == "Dark" else "Dark"
            ctk.set_appearance_mode(new_theme)
            self.settings['ui_settings']['theme'] = new_theme.lower()
            self.save_settings()
        except Exception as e:
            self.show_error(f"Error toggling theme: {e}")

    def save_settings(self):
        """Save current settings to file and record history."""
        try:
            errs = validate_settings(self.settings)
            if errs:
                self.show_error(f"Cannot save settings due to validation errors: {errs}")
                return
            settings_path = os.path.join(CACHE_DIR, 'settings.json')
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(settings_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            record_settings_history(self.settings)
            self.log_to_console("Settings saved")
        except Exception as e:
            logging.error(f"Error saving settings: {e}")

    def load_theme_preference(self):
        """Load and apply saved theme preference."""
        try:
            theme = self.settings.get('ui_settings', {}).get('theme', 'dark')
            ctk.set_appearance_mode(theme.capitalize())
        except Exception as e:
            logging.error(f"Error loading theme preference: {e}")
            ctk.set_appearance_mode("Dark")

    def update_citation_format(self, format_name: str):
        """Update citation format and regenerate bibliography."""
        try:
            self.settings['ui_settings']['citation_format'] = format_name
            self.save_settings()
            if self.doc_refiner and self.doc_refiner.bibliography:
                self.update_bibliography()
        except Exception as e:
            self.show_error(f"Error updating citation format: {e}")

    def update_bibliography(self):
        """Update bibliography text with current format."""
        try:
            bib_text = self.doc_refiner.generate_bibliography_text()
            self.bib_textbox.configure(state="normal")
            self.bib_textbox.delete("1.0", "end")
            self.bib_textbox.insert("end", bib_text)
            self.bib_textbox.configure(state="disabled")
        except Exception as e:
            self.show_error(f"Error updating bibliography: {e}")

    def create_control_frame(self):
        """Create the control frame with buttons and options."""
        control_frame = ctk.CTkFrame(self.root)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        # Create and store button references for tooltips
        self.upload_button = ctk.CTkButton(control_frame, text="Upload Document", 
                                         command=self.upload_document)
        self.upload_button.pack(side="left", padx=5)
        
        self.refine_button = ctk.CTkButton(control_frame, text="Refine & Add Citations", 
                                          command=self.start_refinement,
                                          state="disabled")  # Initially disabled
        self.refine_button.pack(side="left", padx=5)
        
        self.save_button = ctk.CTkButton(control_frame, text="Save Refined Document", 
                                        command=self.save_refined_document,
                                        state="disabled")  # Initially disabled
        self.save_button.pack(side="left", padx=5)
        
        self.bib_button = ctk.CTkButton(control_frame, text="Save Bibliography", 
                                       command=self.save_bibliography,
                                       state="disabled")  # Initially disabled
        self.bib_button.pack(side="left", padx=5)
        
        # Theme toggle
        self.theme_button = ctk.CTkButton(control_frame, text="Toggle Theme",
                                        command=self.toggle_theme)
        self.theme_button.pack(side="right", padx=5)
        
        # Citation format selector
        self.citation_format = ctk.CTkComboBox(
            control_frame,
            values=list(CITATION_FORMATS.keys()),
            command=self.update_citation_format,
            state="readonly"  # Prevent manual editing
        )
        self.citation_format.pack(side="right", padx=5)
        self.citation_format.set(self.settings.get('ui_settings', {}).get('citation_format', 'APA'))
        
    def create_progress_frame(self):
        """Create the progress tracking frame."""
        progress_frame = ctk.CTkFrame(self.root)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(progress_frame, text="Ready")
        self.status_label.pack(pady=5)
        
    def create_notebook_frame(self):
        """Create the notebook with document tabs."""
        self.notebook = ctk.CTkTabview(self.root, width=950, height=500)
        self.notebook.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Add tabs
        self.notebook.add("Original Document")
        self.notebook.add("Refined Document")
        self.notebook.add("Bibliography")
        
        # Create text boxes
        self.orig_textbox = ctk.CTkTextbox(self.notebook.tab("Original Document"))
        self.orig_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.refined_textbox = ctk.CTkTextbox(self.notebook.tab("Refined Document"))
        self.refined_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.bib_textbox = ctk.CTkTextbox(self.notebook.tab("Bibliography"))
        self.bib_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_console_frame(self):
        """Create the console output frame."""
        console_frame = ctk.CTkFrame(self.root)
        console_frame.pack(fill="x", padx=10, pady=5)
        
        self.console = ctk.CTkTextbox(console_frame, height=100)
        self.console.pack(fill="x", padx=5, pady=5)
        
    def setup_keyboard_shortcuts(self):
        self.root.bind('<Control-o>', lambda e: self.upload_document())
        self.root.bind('<Control-s>', lambda e: self.save_refined_document())
        self.root.bind('<Control-b>', lambda e: self.save_bibliography())
        self.root.bind('<Control-r>', lambda e: self.start_refinement())
        
    def start_refinement(self, event=None):
        """Non-blocking wrapper for async refinement."""
        if not self.async_loop:
            self.show_error("Async loop not initialized")
            return
            
        if not self.document_text:
            self.show_error("Please upload a document first")
            return
            
        # Disable buttons during processing
        self.refine_button.configure(state="disabled")
        self.upload_button.configure(state="disabled")
        
        # Start async processing
        asyncio.run_coroutine_threadsafe(
            self.start_refinement_async(),
            self.async_loop
        )
        
    async def start_refinement_async(self):
        """Asynchronous document refinement with proper error handling."""
        if not self.document_text:
            self.show_error("Please upload a document first.")
            return
            
        self.log_to_console("Starting document refinement...")
        self.update_progress(0, "Initializing refinement...")
        
        try:
            # Process document
            self.refined_text = await self.doc_refiner.refine_document(self.document_text)
            
            # Update UI with results
            self.update_refined_text()
            self.update_bibliography()
            
            # Update status
            self.update_progress(1.0, "Refinement complete")
            self.log_to_console("Document refinement complete.")
            
        except Exception as e:
            self.show_error(f"Refinement failed: {str(e)}")
            self.update_progress(0, "Refinement failed")
        finally:
            # Re-enable buttons
            self.refine_button.configure(state="normal")
            self.upload_button.configure(state="normal")
            
    def update_refined_text(self):
        """Update the refined text display."""
        try:
            self.refined_textbox.configure(state="normal")
            self.refined_textbox.delete("1.0", "end")
            self.refined_textbox.insert("end", self.refined_text)
            self.refined_textbox.configure(state="disabled")
            
            # Enable save buttons
            self.save_button.configure(state="normal")
            self.bib_button.configure(state="normal")
        except Exception as e:
            self.show_error(f"Error updating refined text: {e}")

    def create_control_frame(self):
        """Create the control frame with buttons and options."""
        self.control_frame = ctk.CTkFrame(self.root)
        self.control_frame.pack(fill="x", padx=10, pady=5)
        
        # Upload button
        self.upload_button = ctk.CTkButton(
            self.control_frame, 
            text="Upload Document", 
            command=self.upload_document,
            width=120
        )
        self.upload_button.pack(side="left", padx=5)
        
        # Refine button
        self.refine_button = ctk.CTkButton(
            self.control_frame, 
            text="Refine Document", 
            command=self.start_refinement,
            state="disabled",
            width=120
        )
        self.refine_button.pack(side="left", padx=5)
        
        # Save button
        self.save_button = ctk.CTkButton(
            self.control_frame, 
            text="Save Document", 
            command=self.save_document,
            state="disabled",
            width=120
        )
        self.save_button.pack(side="left", padx=5)
        
        # Bibliography button
        self.bib_button = ctk.CTkButton(
            self.control_frame, 
            text="Save Bibliography", 
            command=self.save_bibliography,
            state="disabled",
            width=120
        )
        self.bib_button.pack(side="left", padx=5)
        
        # Theme toggle button
        self.theme_button = ctk.CTkButton(
            self.control_frame, 
            text="Toggle Theme", 
            command=self.toggle_theme,
            width=100
        )
        self.theme_button.pack(side="right", padx=5)
        
        # Citation format dropdown
        self.citation_format = ctk.CTkOptionMenu(
            self.control_frame,
            values=list(CITATION_FORMATS.keys()),
            command=self.change_citation_format
        )
        self.citation_format.set(self.settings.get('ui_settings', {}).get('citation_format', 'APA'))
        self.citation_format.pack(side="right", padx=5)

    def create_progress_frame(self):
        """Create the progress frame with progress bar and status."""
        self.progress_frame = ctk.CTkFrame(self.root)
        self.progress_frame.pack(fill="x", padx=10, pady=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5)
        self.progress_bar.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(self.progress_frame, text="Ready")
        self.status_label.pack(side="right", padx=5)

    def create_notebook_frame(self):
        """Create the notebook frame with text areas."""
        self.notebook_frame = ctk.CTkFrame(self.root)
        self.notebook_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self.notebook_frame)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Original document tab
        self.tabview.add("Original Document")
        self.orig_textbox = ctk.CTkTextbox(
            self.tabview.tab("Original Document"),
            wrap="word",
            font=("Consolas", 12)
        )
        self.orig_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Refined document tab
        self.tabview.add("Refined Document")
        self.refined_textbox = ctk.CTkTextbox(
            self.tabview.tab("Refined Document"),
            wrap="word",
            font=("Consolas", 12)
        )
        self.refined_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bibliography tab
        self.tabview.add("Bibliography")
        self.bib_textbox = ctk.CTkTextbox(
            self.tabview.tab("Bibliography"),
            wrap="word",
            font=("Consolas", 12)
        )
        self.bib_textbox.pack(fill="both", expand=True, padx=5, pady=5)

        # Settings tab
        self.create_settings_tab()

        # Relevant sources tab
        self.create_sources_tab()

    def create_sources_tab(self):
        self.tabview.add("Relevant Sources")
        tab = self.tabview.tab("Relevant Sources")
        container = ctk.CTkFrame(tab)
        container.pack(fill="both", expand=True, padx=5, pady=5)

        input_label = ctk.CTkLabel(container, text="Document Content")
        input_label.pack(anchor="w")
        self.sources_input_text = ctk.CTkTextbox(container, wrap="word", height=180)
        self.sources_input_text.pack(fill="x", padx=5, pady=5)

        params_frame = ctk.CTkFrame(container)
        params_frame.pack(fill="x", padx=5, pady=5)

        rows_label = ctk.CTkLabel(params_frame, text="Max Results")
        rows_label.pack(side="left", padx=(5, 2))
        self.sources_rows = ctk.CTkComboBox(params_frame, values=["50", "100", "200", "500"])
        self.sources_rows.pack(side="left", padx=5)
        self.sources_rows.set(str(self.settings.get('search_settings', {}).get('max_results', 500)))

        thr_label = ctk.CTkLabel(params_frame, text="Similarity Threshold")
        thr_label.pack(side="left", padx=(15, 2))
        self.sources_thr = ctk.CTkEntry(params_frame, width=80)
        self.sources_thr.pack(side="left", padx=5)
        self.sources_thr.insert(0, str(self.settings.get('similarity_threshold', 0.7)))

        self.sources_use_refine = ctk.CTkSwitch(params_frame, text="Use Refined Query")
        self.sources_use_refine.pack(side="left", padx=15)
        self.sources_use_refine.select()

        run_frame = ctk.CTkFrame(container)
        run_frame.pack(fill="x", padx=5, pady=5)
        self.sources_run_button = ctk.CTkButton(run_frame, text="Find Top Sources", command=self.start_source_ranking)
        self.sources_run_button.pack(side="left")
        self.sources_export_bib_btn = ctk.CTkButton(run_frame, text="Download as .bib", command=self.export_sources_bib)
        self.sources_export_bib_btn.pack(side="left", padx=8)
        try:
            ctk.CTkToolTip(self.sources_export_bib_btn, message="Export current sources in BibTeX format")
        except Exception:
            pass

        tree_frame = ctk.CTkFrame(container)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        columns = ("Title", "Authors", "Year", "Journal", "DOI", "Score", "Summary", "Link")
        self.sources_tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        for col in columns:
            self.sources_tree.heading(col, text=col)
            self.sources_tree.column(col, width=120 if col not in ("Title", "Summary") else 260, stretch=True)
        self.sources_tree.pack(fill="both", expand=True)
        self.sources_tree.bind("<Double-1>", self._open_selected_link)

    def export_sources_bib(self):
        try:
            if not getattr(self, 'sources_results', None):
                self.show_error("No sources to export. Run 'Find Top Sources' first.")
                return
            bib = generate_bibtex_from_sources(self.sources_results)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = filedialog.asksaveasfilename(
                title="Save Sources as BibTeX",
                initialfile=f"sources_{ts}.bib",
                defaultextension=".bib",
                filetypes=[["BibTeX Files","*.bib"],["All Files","*.*"]]
            )
            if not filename:
                return
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(bib)
            self.log_to_console(f"Saved BibTeX: {filename}")
        except Exception as e:
            self.show_error(f"BibTeX export failed: {e}")

    def _create_slider_with_entry(self, parent, label_text: str, setting_keys: List[str], from_: float, to: float, steps: int, type_conv=int, tooltip=""):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=2)
        
        # Label on top or left? Let's do top left for label, and row below for slider + entry
        # Actually, let's do: Label (left), Entry (right), Slider (below full width) OR
        # Label (top left), Slider (left), Entry (right)
        
        # Implementation:
        # Row 1: Label
        # Row 2: Slider (expand) | Entry (fixed width)
        
        ctk.CTkLabel(frame, text=label_text).pack(anchor="w", padx=0)
        
        row2 = ctk.CTkFrame(frame, fg_color="transparent")
        row2.pack(fill="x", pady=(0, 2))
        
        # Get current value
        current_val = self.settings
        for k in setting_keys:
            current_val = current_val.get(k, {})
        if isinstance(current_val, dict): # Fallback if key missing
            current_val = from_
            
        var = ctk.StringVar(value=str(current_val))
        
        def update_from_slider(v):
            val = float(v)
            if type_conv is int:
                val = int(round(val))
            else:
                val = round(val, 2)
            var.set(str(val))
            self._update_setting(setting_keys, val)
            
        def update_from_entry(event=None):
            try:
                val = float(var.get())
                # Clamp value
                val = max(from_, min(to, val))
                if type_conv is int:
                    val = int(round(val))
                else:
                    val = round(val, 2)
                
                # Update slider without triggering callback loop if possible, 
                # but ctk slider .set doesn't trigger command, so it's safe.
                slider.set(val)
                var.set(str(val)) # Format it back
                self._update_setting(setting_keys, val)
            except ValueError:
                pass # Ignore invalid input
        
        slider = ctk.CTkSlider(row2, from_=from_, to=to, number_of_steps=steps, command=update_from_slider)
        slider.set(current_val)
        slider.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        entry = ctk.CTkEntry(row2, textvariable=var, width=60)
        entry.pack(side="right")
        entry.bind("<FocusOut>", update_from_entry)
        entry.bind("<Return>", update_from_entry)
        
        if tooltip:
            try:
                ctk.CTkToolTip(slider, message=tooltip)
                ctk.CTkToolTip(entry, message=tooltip)
            except Exception:
                pass

    def _create_collapsible_section(self, parent, title: str):
        container = ctk.CTkFrame(parent)
        container.pack(fill="x", padx=5, pady=5)
        header = ctk.CTkFrame(container)
        header.pack(fill="x")
        label = ctk.CTkLabel(header, text=title)
        label.pack(side="left", padx=5)
        states = self.settings.get('ui_settings', {}).get('section_states', {})
        btn = ctk.CTkButton(header, text="Hide" if states.get(title, True) else "Show", width=60)
        btn.pack(side="right", padx=5)
        content = ctk.CTkFrame(container)
        content.pack(fill="x", padx=4, pady=4)

        def toggle():
            self.settings.setdefault('ui_settings', {}).setdefault('section_states', {})
            state = self.settings['ui_settings']['section_states'].get(title, True)
            new_state = not state
            self.settings['ui_settings']['section_states'][title] = new_state
            if new_state:
                content.pack(fill="x", padx=4, pady=4)
                btn.configure(text="Hide")
            else:
                content.pack_forget()
                btn.configure(text="Show")
            self.settings_dirty = True
        btn.configure(command=toggle)

        # initialize according to state
        if not states.get(title, True):
            content.pack_forget()
            btn.configure(text="Show")
        return container, content

    def create_settings_tab(self):
        self.tabview.add("Settings")
        tab = self.tabview.tab("Settings")
        outer = ctk.CTkScrollableFrame(tab)
        outer.pack(fill="both", expand=True, padx=5, pady=5)

        # Top bar: search + actions
        top = ctk.CTkFrame(outer)
        top.pack(fill="x", padx=5, pady=(5, 10))
        ctk.CTkLabel(top, text="Search Settings").pack(side="left", padx=(5, 2))
        self.settings_search = ctk.CTkEntry(top, placeholder_text="Type to filter...", width=240)
        self.settings_search.pack(side="left", padx=5)
        self.settings_search.bind("<KeyRelease>", self._on_search_settings)
        self.unsaved_label = ctk.CTkLabel(top, text="Unsaved changes", text_color="#d9534f")
        self.unsaved_label.pack(side="left", padx=10)
        self.unsaved_label.pack_forget()
        self.settings_dirty = False
        self.apply_btn = ctk.CTkButton(top, text="Apply", command=self._apply_settings)
        self.apply_btn.pack(side="right", padx=5)
        self.export_btn = ctk.CTkButton(top, text="Export", command=self._export_settings)
        self.export_btn.pack(side="right", padx=5)
        self.import_btn = ctk.CTkButton(top, text="Import", command=self._import_settings)
        self.import_btn.pack(side="right", padx=5)
        self.save_btn = ctk.CTkButton(top, text="Save", command=self.save_settings)
        self.save_btn.pack(side="right", padx=5)

        self._settings_sections = {}

        # Appearance
        app_container, app_content = self._create_collapsible_section(outer, "Appearance")
        self._settings_sections['Appearance'] = app_container
        ctk.CTkLabel(app_content, text="Theme").pack(anchor="w", padx=5)
        theme_menu = ctk.CTkOptionMenu(app_content, values=["dark", "light"], command=lambda v: self._update_setting(['ui_settings','theme'], v))
        theme_menu.pack(fill="x", padx=5, pady=2)
        theme_menu.set(self.settings['ui_settings'].get('theme', 'dark'))
        try:
            ctk.CTkToolTip(theme_menu, message="Switch between dark and light themes")
        except Exception:
            pass

        ctk.CTkLabel(app_content, text="Citation Format").pack(anchor="w", padx=5)
        cite_menu = ctk.CTkOptionMenu(app_content, values=list(CITATION_FORMATS.keys()), command=lambda v: self._update_setting(['ui_settings','citation_format'], v))
        cite_menu.pack(fill="x", padx=5, pady=2)
        cite_menu.set(self.settings['ui_settings'].get('citation_format', 'APA'))
        try:
            ctk.CTkToolTip(cite_menu, message="Choose bibliography formatting")
        except Exception:
            pass

        ctk.CTkLabel(app_content, text="Window Size").pack(anchor="w", padx=5)
        ws_entry = ctk.CTkEntry(app_content)
        ws_entry.pack(fill="x", padx=5, pady=2)
        ws_entry.insert(0, self.settings['ui_settings'].get('window_size', '1000x750'))
        ws_entry.bind("<FocusOut>", lambda e: self._update_setting(['ui_settings','window_size'], ws_entry.get()))

        ctk.CTkLabel(app_content, text="Accent Color").pack(anchor="w", padx=5)
        color_btn = ctk.CTkButton(app_content, text="Pick Color", command=lambda: self._pick_color(['ui_settings','accent_color']))
        color_btn.pack(padx=5, pady=2)

        reset_app = ctk.CTkButton(app_content, text="Reset Appearance to Defaults", command=lambda: self._reset_section('ui_settings'))
        reset_app.pack(padx=5, pady=(8, 2))

        # Performance
        perf_container, perf_content = self._create_collapsible_section(outer, "Performance")
        self._settings_sections['Performance'] = perf_container
        ctk.CTkLabel(perf_content, text="Summarizer Model").pack(anchor="w", padx=5)
        sum_menu = ctk.CTkEntry(perf_content)
        sum_menu.pack(fill="x", padx=5, pady=2)
        sum_menu.insert(0, self.settings['model_settings'].get('summarizer_model', 'facebook/bart-large-cnn'))
        sum_menu.bind("<FocusOut>", lambda e: self._update_setting(['model_settings','summarizer_model'], sum_menu.get()))

        ctk.CTkLabel(perf_content, text="Sentence Model").pack(anchor="w", padx=5)
        sen_entry = ctk.CTkEntry(perf_content)
        sen_entry.pack(fill="x", padx=5, pady=2)
        sen_entry.insert(0, self.settings['model_settings'].get('sentence_model', 'all-MiniLM-L6-v2'))
        sen_entry.bind("<FocusOut>", lambda e: self._update_setting(['model_settings','sentence_model'], sen_entry.get()))

        self._create_slider_with_entry(perf_content, "Max Summary Length", ['model_settings','max_length'], 2, 200, 198, int)
        self._create_slider_with_entry(perf_content, "Min Summary Length", ['model_settings','min_length'], 1, 100, 99, int)
        self._create_slider_with_entry(perf_content, "Similarity Threshold", ['similarity_threshold'], 0.0, 1.0, 100, float, "Minimum similarity score (0-1)")
        self._create_slider_with_entry(perf_content, "Max Concurrency", ['async_settings','max_concurrency'], 1, 16, 15, int, "Max concurrent tasks")

        reset_perf = ctk.CTkButton(perf_content, text="Reset Performance to Defaults", command=lambda: self._reset_section('model_settings'))
        reset_perf.pack(padx=5, pady=(8, 2))

        # Search & Ranking
        rank_container, rank_content = self._create_collapsible_section(outer, "Search & Ranking")
        self._settings_sections['Search & Ranking'] = rank_container
        
        self._create_slider_with_entry(rank_content, "Max Search Results", ['search_settings','max_results'], 10, 1000, 99, int, "Max papers to fetch")
        
        # Snowballing
        snow_sw = ctk.CTkSwitch(rank_content, text="Enable Snowballing")
        snow_sw.pack(anchor="w", padx=5, pady=2)
        if self.settings['search_settings'].get('enable_snowballing', False):
            snow_sw.select()
        snow_sw.configure(command=lambda: self._update_setting(['search_settings','enable_snowballing'], bool(snow_sw.get())))
        
        self._create_slider_with_entry(rank_content, "Snowballing Depth", ['search_settings','snowballing_depth'], 1, 3, 2, int)

        ctk.CTkLabel(rank_content, text="Ranking Weights").pack(anchor="w", padx=5, pady=(10,2))
        
        self._create_slider_with_entry(rank_content, "Similarity Weight", ['ranking_settings','weight_similarity'], 0.0, 1.0, 100, float)
        self._create_slider_with_entry(rank_content, "Overlap Weight", ['ranking_settings','weight_overlap'], 0.0, 1.0, 100, float)
        self._create_slider_with_entry(rank_content, "Citation Weight", ['ranking_settings','weight_citations'], 0.0, 1.0, 100, float)
        self._create_slider_with_entry(rank_content, "Recency Weight", ['ranking_settings','weight_recency'], 0.0, 1.0, 100, float)
        
        reset_rank = ctk.CTkButton(rank_content, text="Reset Search & Ranking to Defaults", command=lambda: (self._reset_section('search_settings'), self._reset_section('ranking_settings')))
        reset_rank.pack(padx=5, pady=(8, 2))

        # Privacy
        priv_container, priv_content = self._create_collapsible_section(outer, "Privacy")
        self._settings_sections['Privacy'] = priv_container
        telemetry_sw = ctk.CTkSwitch(priv_content, text="Telemetry Enabled")
        telemetry_sw.pack(anchor="w", padx=5, pady=2)
        if self.settings['privacy_settings'].get('telemetry_enabled', False):
            telemetry_sw.select()
        telemetry_sw.configure(command=lambda: self._update_setting(['privacy_settings','telemetry_enabled'], bool(telemetry_sw.get())))

        cache_sw = ctk.CTkSwitch(priv_content, text="Cache Enabled")
        cache_sw.pack(anchor="w", padx=5, pady=2)
        if self.settings['privacy_settings'].get('cache_enabled', True):
            cache_sw.select()
        cache_sw.configure(command=lambda: self._update_setting(['privacy_settings','cache_enabled'], bool(cache_sw.get())))

        clear_cache_btn = ctk.CTkButton(priv_content, text="Clear Cache", command=self._clear_cache)
        clear_cache_btn.pack(padx=5, pady=(4, 2))

        reset_priv = ctk.CTkButton(priv_content, text="Reset Privacy to Defaults", command=lambda: self._reset_section('privacy_settings'))
        reset_priv.pack(padx=5, pady=(8, 2))

        # Notifications
        notif_container, notif_content = self._create_collapsible_section(outer, "Notifications")
        self._settings_sections['Notifications'] = notif_container
        sound_sw = ctk.CTkSwitch(notif_content, text="Sound Enabled")
        sound_sw.pack(anchor="w", padx=5, pady=2)
        if self.settings['notification_settings'].get('sound_enabled', False):
            sound_sw.select()
        sound_sw.configure(command=lambda: self._update_setting(['notification_settings','sound_enabled'], bool(sound_sw.get())))

        desk_sw = ctk.CTkSwitch(notif_content, text="Desktop Notifications")
        desk_sw.pack(anchor="w", padx=5, pady=2)
        if self.settings['notification_settings'].get('desktop_notifications', False):
            desk_sw.select()
        desk_sw.configure(command=lambda: self._update_setting(['notification_settings','desktop_notifications'], bool(desk_sw.get())))

        ctk.CTkLabel(notif_content, text="Verbosity").pack(anchor="w", padx=5)
        verb_menu = ctk.CTkOptionMenu(notif_content, values=["quiet","normal","verbose"], command=lambda v: self._update_setting(['notification_settings','verbosity'], v))
        verb_menu.set(self.settings['notification_settings'].get('verbosity','normal'))
        verb_menu.pack(fill="x", padx=5, pady=2)

        reset_notif = ctk.CTkButton(notif_content, text="Reset Notifications to Defaults", command=lambda: self._reset_section('notification_settings'))
        reset_notif.pack(padx=5, pady=(8, 2))

        # Developer
        dev_container, dev_content = self._create_collapsible_section(outer, "Developer")
        self._settings_sections['Developer'] = dev_container
        dev_sw = ctk.CTkSwitch(dev_content, text="Developer Mode")
        dev_sw.pack(anchor="w", padx=5, pady=2)
        if self.settings['developer_settings'].get('developer_mode', False):
            dev_sw.select()
        dev_sw.configure(command=lambda: self._toggle_developer_mode(bool(dev_sw.get())))

        ctk.CTkLabel(dev_content, text="Log Level").pack(anchor="w", padx=5)
        log_menu = ctk.CTkOptionMenu(dev_content, values=["DEBUG","INFO","WARNING","ERROR"], command=lambda v: self._update_log_level(v))
        log_menu.set(self.settings['developer_settings'].get('log_level','INFO'))
        log_menu.pack(fill="x", padx=5, pady=2)

        hist_btn = ctk.CTkButton(dev_content, text="Revert to Previous Settings", command=self._revert_last_settings)
        hist_btn.pack(padx=5, pady=(8, 2))

        reset_dev = ctk.CTkButton(dev_content, text="Reset Developer to Defaults", command=lambda: self._reset_section('developer_settings'))
        reset_dev.pack(padx=5, pady=(8, 2))

    def _update_setting(self, path_keys: List[str], value):
        try:
            target = self.settings
            for k in path_keys[:-1]:
                target = target.setdefault(k, {})
            target[path_keys[-1]] = value
            self.settings_dirty = True
            try:
                self.unsaved_label.pack(side="left", padx=10)
            except Exception:
                pass
        except Exception as e:
            self.show_error(f"Failed to update setting {'.'.join(path_keys)}: {e}")

    def _pick_color(self, path_keys: List[str]):
        try:
            _, hex_color = colorchooser.askcolor(title="Choose accent color")
            if hex_color:
                self._update_setting(path_keys, hex_color)
        except Exception as e:
            self.show_error(f"Color pick failed: {e}")

    def _reset_section(self, section_key: str):
        try:
            reset_section_to_defaults(self.settings, section_key, DEFAULT_SETTINGS)
            self.settings_dirty = True
            self.save_settings()
        except Exception as e:
            self.show_error(f"Failed to reset {section_key}: {e}")

    def _on_search_settings(self, event=None):
        try:
            term = self.settings_search.get().strip().lower()
            if not term:
                for name, cont in self._settings_sections.items():
                    cont.pack(fill="x", padx=5, pady=5)
                return
            for name, cont in self._settings_sections.items():
                if term in name.lower():
                    cont.pack(fill="x", padx=5, pady=5)
                else:
                    cont.pack_forget()
        except Exception:
            pass

    def _import_settings(self):
        try:
            filename = filedialog.askopenfilename(title="Import Settings", filetypes=[["JSON Files","*.json"],["All Files","*.*"]])
            if not filename:
                return
            with open(filename, 'r') as f:
                incoming = json.load(f)
            merged = {**DEFAULT_SETTINGS, **incoming}
            errs = validate_settings(merged)
            if errs:
                self.show_error(f"Invalid imported settings: {errs}")
                return
            self.settings = merged
            self.save_settings()
            self.log_to_console(f"Imported settings from {filename}")
        except Exception as e:
            self.show_error(f"Import failed: {e}")

    def _export_settings(self):
        try:
            filename = filedialog.asksaveasfilename(title="Export Settings", defaultextension=".json", filetypes=[["JSON Files","*.json"],["All Files","*.*"]])
            if not filename:
                return
            with open(filename, 'w') as f:
                json.dump(self.settings, f, indent=4)
            self.log_to_console(f"Exported settings to {filename}")
        except Exception as e:
            self.show_error(f"Export failed: {e}")

    def _apply_settings(self):
        try:
            errs = validate_settings(self.settings)
            if errs:
                self.show_error(f"Cannot apply settings due to: {errs}")
                return
            # Apply theme immediately
            theme = self.settings.get('ui_settings', {}).get('theme', 'dark')
            ctk.set_appearance_mode(theme)
            # Apply log level
            self._update_log_level(self.settings.get('developer_settings', {}).get('log_level','INFO'))
            # Apply window size
            ws = self.settings.get('ui_settings', {}).get('window_size','1000x750')
            try:
                self.root.geometry(ws)
            except Exception:
                pass
            # Reinitialize heavy components if needed (models)
            # Deferred: require restart or implement safe reinit; here we log
            self.log_to_console("Applied settings")
            self.settings_dirty = False
            try:
                self.unsaved_label.pack_forget()
            except Exception:
                pass
        except Exception as e:
            self.show_error(f"Apply failed: {e}")

    def _update_log_level(self, level: str):
        try:
            self._update_setting(['developer_settings','log_level'], level)
            logging.getLogger().setLevel(getattr(logging, level, logging.INFO))
        except Exception as e:
            self.show_error(f"Failed to set log level: {e}")

    def _revert_last_settings(self):
        try:
            history = load_settings_history()
            if len(history) < 2:
                self.show_error("No previous settings to revert to")
                return
            prev = history[-2]['settings']
            self.settings = prev
            self.save_settings()
            self.log_to_console("Reverted to previous settings")
        except Exception as e:
            self.show_error(f"Revert failed: {e}")

    def _toggle_developer_mode(self, enabled: bool):
        self._update_setting(['developer_settings','developer_mode'], enabled)
        self.log_to_console(f"Developer mode {'enabled' if enabled else 'disabled'}")

    def _clear_cache(self):
        try:
            # remove all files in CACHE_DIR except settings files
            for name in os.listdir(CACHE_DIR):
                if name in ('settings.json','settings_history.json'):
                    continue
                path = os.path.join(CACHE_DIR, name)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception:
                    pass
            self.log_to_console("Cache cleared")
        except Exception as e:
            self.show_error(f"Failed to clear cache: {e}")

    def _open_selected_link(self, event=None):
        sel = self.sources_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        values = self.sources_tree.item(item_id, 'values')
        if len(values) >= 8 and values[7]:
            try:
                webbrowser.open(values[7])
            except Exception as e:
                self.show_error(f"Failed to open link: {e}")

    def start_source_ranking(self):
        if not self.async_loop:
            self.show_error("Async loop not initialized")
            return
        text = self.sources_input_text.get("1.0", "end").strip()
        if not text:
            self.show_error("Please paste or type document content")
            return
        try:
            rows = int(self.sources_rows.get())
        except Exception:
            rows = self.settings.get('search_settings', {}).get('max_results', 500)
        try:
            thr = float(self.sources_thr.get())
        except Exception:
            thr = float(self.settings.get('similarity_threshold', 0.7))
        use_refine = bool(self.sources_use_refine.get())

        self.sources_run_button.configure(state="disabled")
        self.log_to_console("Starting source ranking...")

        asyncio.run_coroutine_threadsafe(
            self.run_source_ranking_async(text, rows, thr, use_refine),
            self.async_loop
        )

    async def run_source_ranking_async(self, text: str, rows: int, thr: float, use_refine: bool):
        try:
            def progress_cb(val, status):
                self.update_progress(val, status)
            self.update_progress(0.0, "Initializing source ranking...")
            results = self.source_ranking.rank_top_sources(text, rows, thr, use_refine, progress_cb)
            self.update_sources_results(results)
            self.update_progress(1.0, "Source ranking complete")
            self.log_to_console("Relevant sources computed.")
        except Exception as e:
            self.show_error(f"Source ranking failed: {e}")
            self.update_progress(0.0, "Source ranking failed")
        finally:
            self.sources_run_button.configure(state="normal")

    def update_sources_results(self, results: List[Dict[str, Any]]):
        try:
            def build_doi_url(doi: str):
                try:
                    if not doi:
                        return None
                    d = str(doi).strip()
                    if not re.match(r'^10\.\d{4,9}/\S+$', d):
                        return None
                    return f"https://doi.org/{quote(d)}"
                except Exception:
                    return None

            self.sources_results = []
            for row in self.sources_tree.get_children():
                self.sources_tree.delete(row)
            for r in results:
                doi = r.get('doi')
                doi_url = build_doi_url(doi)
                if doi_url:
                    r['link'] = doi_url
                self.sources_results.append(r)
                self.sources_tree.insert('', 'end', values=(
                    r.get('title', ''),
                    r.get('authors', ''),
                    r.get('year', ''),
                    r.get('journal', ''),
                    r.get('doi', ''),
                    f"{r.get('score', 0.0):.3f}",
                    r.get('summary', ''),
                    r.get('link', '')
                ))
        except Exception as e:
            self.show_error(f"Error displaying results: {e}")

    def create_console_frame(self):
        """Create the console frame for logging."""
        self.console_frame = ctk.CTkFrame(self.root, height=150)
        self.console_frame.pack(fill="x", padx=10, pady=5)
        self.console_frame.pack_propagate(False)
        
        # Console label
        console_label = ctk.CTkLabel(self.console_frame, text="Console Output:")
        console_label.pack(anchor="w", padx=5, pady=(5, 0))
        
        # Console textbox
        self.console_textbox = ctk.CTkTextbox(
            self.console_frame,
            height=120,
            font=("Consolas", 10)
        )
        self.console_textbox.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for the application."""
        self.root.bind('<Control-o>', self.upload_document)
        self.root.bind('<Control-r>', lambda e: self.start_refinement() if self.refine_button.cget("state") == "normal" else None)
        self.root.bind('<Control-s>', lambda e: self.save_document() if self.save_button.cget("state") == "normal" else None)
        self.root.bind('<Control-b>', lambda e: self.save_bibliography() if self.bib_button.cget("state") == "normal" else None)

    def load_theme_preference(self):
        """Load theme preference from settings."""
        theme = self.settings.get('ui_settings', {}).get('theme', 'dark')
        ctk.set_appearance_mode(theme)

    def save_document(self, event=None):
        """Save the refined document."""
        if not self.refined_text:
            self.show_error("No refined document to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Refined Document",
                defaultextension=".tex",
                filetypes=[
                    ("LaTeX Files", "*.tex"),
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.refined_text)
                self.log_to_console(f"Saved refined document: {filename}")
        except Exception as e:
            self.show_error(f"Error saving document: {e}")

    def save_bibliography(self, event=None):
        """Save the bibliography."""
        if not self.doc_refiner or not self.doc_refiner.bibliography:
            self.show_error("No bibliography to save.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Bibliography",
                defaultextension=".bib",
                filetypes=[
                    ("BibTeX Files", "*.bib"),
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ]
            )
            if filename:
                bib_text = self.doc_refiner.generate_bibliography_text()
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(bib_text)
                self.log_to_console(f"Saved bibliography: {filename}")
        except Exception as e:
            self.show_error(f"Error saving bibliography: {e}")

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        current_mode = ctk.get_appearance_mode()
        new_mode = "light" if current_mode == "Dark" else "dark"
        ctk.set_appearance_mode(new_mode)
        
        # Update settings
        self.settings['ui_settings']['theme'] = new_mode
        self.log_to_console(f"Switched to {new_mode} theme")

    def change_citation_format(self, format_name):
        """Change the citation format."""
        self.settings['ui_settings']['citation_format'] = format_name
        self.log_to_console(f"Citation format changed to {format_name}")

    def update_progress(self, value, text=""):
        """Update the progress bar and status text."""
        try:
            self.progress_bar.set(value)
            if text:
                self.status_label.configure(text=text)
            self.root.update_idletasks()
        except Exception as e:
            logging.error(f"Error updating progress: {e}")

    def update_bibliography(self):
        """Update the bibliography display."""
        try:
            if self.doc_refiner and self.doc_refiner.bibliography:
                bib_text = self.doc_refiner.generate_bibliography_text()
                self.bib_textbox.configure(state="normal")
                self.bib_textbox.delete("1.0", "end")
                self.bib_textbox.insert("end", bib_text)
                self.bib_textbox.configure(state="disabled")
        except Exception as e:
            self.show_error(f"Error updating bibliography: {e}")

    def log_to_console(self, message):
        """Log a message to the console textbox."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            
            self.console_textbox.configure(state="normal")
            self.console_textbox.insert("end", formatted_message)
            self.console_textbox.see("end")
            self.console_textbox.configure(state="disabled")
            
            # Also log to file
            logging.info(message)
        except Exception as e:
            logging.error(f"Error logging to console: {e}")

    def show_error(self, message):
        """Show an error message to the user."""
        try:
            messagebox.showerror("Error", message)
            self.log_to_console(f"ERROR: {message}")
        except Exception as e:
            logging.error(f"Error showing error message: {e}")
            
    def run(self):
        self.root.mainloop()

# ---------------- Settings Utilities ----------------
def settings_history_path() -> str:
    return os.path.join(CACHE_DIR, 'settings_history.json')

def record_settings_history(settings: dict) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        history_file = settings_history_path()
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'version': settings.get('version', 1),
            'settings': settings
        }
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                try:
                    history = json.load(f)
                except Exception:
                    history = []
        history.append(entry)
        # keep last 50 entries
        history = history[-50:]
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to record settings history: {e}")

def load_settings_history() -> List[Dict[str, Any]]:
    try:
        history_file = settings_history_path()
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load settings history: {e}")
    return []

def reset_section_to_defaults(settings: dict, section: str, defaults: dict) -> dict:
    try:
        if section in defaults:
            settings[section] = json.loads(json.dumps(defaults[section]))
        return settings
    except Exception as e:
        logging.error(f"Failed to reset section '{section}' to defaults: {e}")
        return settings

def validate_settings(settings: dict) -> Dict[str, str]:
    errors: Dict[str, str] = {}
    try:
        thr = settings.get('similarity_threshold', 0.7)
        if not isinstance(thr, (int, float)) or not (0.0 <= float(thr) <= 1.0):
            errors['similarity_threshold'] = 'Must be a number between 0 and 1'

        ss = settings.get('search_settings', {})
        if int(ss.get('max_results', 500)) <= 0:
            errors['search_settings.max_results'] = 'Must be a positive integer'
        if float(ss.get('rate_limit', 1.0)) < 0.0:
            errors['search_settings.rate_limit'] = 'Must be non-negative seconds'

        ms = settings.get('model_settings', {})
        if int(ms.get('max_length', 20)) < int(ms.get('min_length', 2)):
            errors['model_settings.max_length'] = 'Must be >= min_length'

        ui = settings.get('ui_settings', {})
        ws = ui.get('window_size', '1000x750')
        if not re.match(r'^\d{3,5}x\d{3,5}$', str(ws)):
            errors['ui_settings.window_size'] = 'Use WxH (e.g., 1200x800)'
        theme = ui.get('theme', 'dark')
        if theme not in ('dark', 'light'):
            errors['ui_settings.theme'] = 'Theme must be dark or light'

        asyncs = settings.get('async_settings', {})
        if int(asyncs.get('max_concurrency', 4)) <= 0:
            errors['async_settings.max_concurrency'] = 'Must be >= 1'
    except Exception as e:
        logging.error(f"Validation error: {e}")
        errors['__internal__'] = 'Validation failed'
    return errors

def deep_merge_settings(defaults: dict, custom: dict) -> dict:
    try:
        if not isinstance(defaults, dict) or not isinstance(custom, dict):
            return custom if custom is not None else defaults
        merged = json.loads(json.dumps(defaults))
        for k, v in custom.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = deep_merge_settings(merged[k], v)
            else:
                merged[k] = v
        return merged
    except Exception as e:
        logging.warning(f"Deep merge failed: {e}")
        return custom

def escape_bibtex(value: str) -> str:
    try:
        s = str(value)
        s = s.replace('\\', '\\')
        for ch, rep in {
            '{': '\\{',
            '}': '\\}',
            '%': '\\%',
            '_': '\\_',
            '&': '\\&',
            '#': '\\#',
            '~': '{\\textasciitilde}'
        }.items():
            s = s.replace(ch, rep)
        return s
    except Exception:
        return str(value)

def generate_bibtex_from_sources(results: List[Dict[str, Any]]) -> str:
    entries = []
    for i, r in enumerate(results, start=1):
        typ = 'article'
        title = escape_bibtex(r.get('title', f'Untitled {i}'))
        authors = escape_bibtex(r.get('authors', ''))
        journal = escape_bibtex(r.get('journal', ''))
        year = escape_bibtex(r.get('year', ''))
        volume = escape_bibtex(r.get('volume', ''))
        number = escape_bibtex(r.get('issue', ''))
        pages = escape_bibtex(r.get('pages', ''))
        doi = str(r.get('doi', '')).strip()
        url = r.get('link', '')
        key_base = (authors.split(',')[0] or 'key').replace(' ', '')
        key_year = year if year else ''
        key = f"{key_base}{key_year}entry{i}"
        lines = [f"@{typ}{{{key},"]
        lines.append(f"  title = {{{{{title}}}}},")
        if authors:
            lines.append(f"  author = {{{authors}}},")
        if journal:
            lines.append(f"  journal = {{{journal}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        if volume:
            lines.append(f"  volume = {{{volume}}},")
        if number:
            lines.append(f"  number = {{{number}}},")
        if pages:
            lines.append(f"  pages = {{{pages}}},")
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        if url:
            lines.append(f"  url = {{{escape_bibtex(url)}}},")
        # remove trailing comma from last field
        if lines[-1].endswith(','):
            lines[-1] = lines[-1][:-1]
        lines.append('}')
        entries.append('\n'.join(lines))
    return '\n\n'.join(entries) + '\n'

# ---------------- Default Settings ----------------
DEFAULT_SETTINGS = {
    'version': 1,
    'similarity_threshold': 0.7,
    'keyword_settings': {
        'lan': 'en',
        'n': 2,
        'dedupLim': 0.9,
        'top': 10
    },
    'search_settings': {
        'max_results': 500,
        'rate_limit': 1.0,
        'enable_snowballing': False,
        'snowballing_depth': 1
    },
    'ranking_settings': {
        'weight_similarity': 0.65,
        'weight_overlap': 0.1,
        'weight_citations': 0.15,
        'weight_recency': 0.1
    },
    'model_settings': {
        'sentence_model': 'all-MiniLM-L6-v2',
        'summarizer_model': 'sshleifer/distilbart-cnn-12-6',
        'max_length': 20,
        'min_length': 2
    },
    'async_settings': {
        'max_concurrency': 4
    },
    'privacy_settings': {
        'telemetry_enabled': False,
        'cache_enabled': True
    },
    'notification_settings': {
        'sound_enabled': False,
        'desktop_notifications': False,
        'verbosity': 'normal'
    },
    'developer_settings': {
        'developer_mode': False,
        'log_level': 'INFO',
        'mock_apis': False
    },
    'ui_settings': {
        'theme': 'dark',
        'citation_format': 'Harvard',
        'window_size': '1000x750',
        'accent_color': '#1f6aa5',
        'section_states': {
            'Appearance': True,
            'Performance': True,
            'Privacy': True,
            'Notifications': True,
            'Developer': False
        }
    }
}

# ---------------- Main ----------------
def main():
    try:
        # Load settings from file if exists, otherwise use defaults
        settings_path = os.path.join(CACHE_DIR, 'settings.json')
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    loaded = json.load(f)
                    settings = deep_merge_settings(DEFAULT_SETTINGS, loaded)
            except Exception as e:
                logging.warning(f"Error loading settings, using defaults: {e}")
                settings = DEFAULT_SETTINGS
        else:
            settings = DEFAULT_SETTINGS
            
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Initialize and run the application
        app = DeepResearchToolApp(settings)
        app.run()
        
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        if 'app' in locals() and hasattr(app, 'root'):
            app.root.destroy()
        messagebox.showerror("Critical Error", f"Application failed to start: {e}")
        raise

def generate_thesis_ideas(thesis_name: str, research_field: str = "Applied Machine Learning", output_path: str = "thesis_proposal.txt", seed: int = 42) -> str:
    import random
    from datetime import datetime
    random.seed(seed)
    aspects = [
        "fairness","explainability","privacy","robustness","causality","uncertainty quantification",
        "calibration","drift detection","domain adaptation","semi-supervised learning","active learning",
        "few-shot learning","multimodal fusion","graph learning","time-series forecasting","reinforcement learning",
        "human-in-the-loop","federated optimization","energy efficiency","scalable training","benchmarking",
        "reproducibility","simulation-to-real","synthetic data generation","annotation efficiency"
    ]
    methods = [
        "graph neural networks","causal inference","reinforcement learning","federated learning",
        "transformers","Bayesian modeling","self-supervised learning","contrastive learning",
        "probabilistic programming","meta-learning","diffusion models"
    ]
    data_sources = [
        "clinical records","satellite imagery","transaction graphs","multimodal sensors",
        "text corpora","time-series logs","knowledge graphs","speech and dialog data",
        "edge-collected datasets","privacy-sensitive user telemetry"
    ]
    contexts = [
        "healthcare","finance","education","climate science","manufacturing",
        "cybersecurity","transportation","e-commerce","public policy","scientific discovery"
    ]
    titles = []
    seen = set()
    templates = [
        "{aspect} in {field} via {method} using {data} for {context}",
        "Towards {aspect} in {field}: {method} on {data} for {context}",
        "{method} for {aspect} with {data} in {context}"
    ]
    combos = [(a, m, d, c) for a in aspects for m in methods for d in data_sources for c in contexts]
    random.shuffle(combos)
    i = 0
    while len(titles) < 100 and i < len(combos):
        a, m, d, c = combos[i]
        tpl = random.choice(templates)
        t = tpl.format(aspect=a.title(), field=research_field, method=m.title(), data=d, context=c.title())
        if t not in seen:
            seen.add(t)
            titles.append(t)
        i += 1
    intro = (
        f"{thesis_name} advances {research_field} by delivering a rigorous, deployable research product that bridges methodological innovation and real-world impact. "
        "The work positions modern learning techniques within a responsible framework, emphasizing transparency, reliability, and stakeholder trust."
    )
    problem = (
        "Despite rapid progress in machine learning, many systems remain difficult to explain, fragile under distribution shift, and constrained by privacy and compliance demands. "
        "Organizations need validated approaches that balance accuracy with governance, enabling safe adoption in high-stakes domains."
    )
    solution = (
        "The proposed solution integrates transformers and graph-based models with causal analysis, uncertainty estimation, and federated optimization to produce robust, privacy-preserving pipelines. "
        "Methodology includes dataset curation, reproducible training, ablation studies, calibration, and comprehensive evaluation across benchmark tasks (Doshi-Velez & Kim, 2017; Kairouz et al., 2021)."
    )
    significance = (
        "Expected outcomes include state-of-the-art performance with interpretable outputs, measurable fairness improvements, reduced drift, and energy-aware training. "
        "The product contributes open evaluation artifacts, deployment guides, and policy-aligned reporting, supporting researchers and practitioners across sectors."
    )
    paragraphs = [intro, problem, solution, significance]
    words = sum(len(p.split()) for p in paragraphs)
    if words < 300:
        extra = (
            "The evaluation plan covers cross-domain generalization, stress testing under realistic noise, and longitudinal monitoring with human-in-the-loop validation. "
            "Results are presented with confidence intervals, sensitivity analyses, and error taxonomies to inform decision makers and align with academic best practices."
        )
        paragraphs.append(extra)
    summary_text = "\n\n".join(paragraphs)
    meta = [
        "Thesis Proposal Generation",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Research Field: {research_field}",
        f"Thesis Name: {thesis_name}"
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta) + "\n\n")
        f.write("=== Thesis Title Ideas (100) ===\n")
    for n, t in enumerate(titles, 1):
        f.write(f"{n}. {t}\n")
        f.write("\n=== Product Summary (300 words) ===\n\n")
        f.write(summary_text)
    return output_path


class TestSettingsValidation(unittest.TestCase):
    def test_default_settings_valid(self):
        errs = validate_settings(DEFAULT_SETTINGS)
        self.assertEqual(errs, {})

    def test_invalid_similarity_threshold(self):
        s = json.loads(json.dumps(DEFAULT_SETTINGS))
        s['similarity_threshold'] = 2.0
        errs = validate_settings(s)
        self.assertIn('similarity_threshold', errs)


class TestSettingsHistory(unittest.TestCase):
    def test_record_and_load_history(self):
        s = json.loads(json.dumps(DEFAULT_SETTINGS))
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_file = os.path.join(tmpdir, 'settings_history.json')
            with patch.object(sys.modules[__name__], 'settings_history_path', return_value=hist_file):
                record_settings_history(s)
                s['version'] = 2
                record_settings_history(s)
                hist = load_settings_history()
                self.assertEqual(len(hist), 2)
                self.assertEqual(hist[-1]['settings']['version'], 2)


def run_internal_tests() -> bool:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSettingsValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestSettingsHistory))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Deep Research Tool GUI or run maintenance.")
    parser.add_argument("--run-tests", action="store_true", help="Run internal unit tests instead of launching the GUI.")
    args = parser.parse_args()
    if args.run_tests:
        success = run_internal_tests()
        sys.exit(0 if success else 1)
    main()
    
