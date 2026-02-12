import React, { useMemo, useState } from 'react';
import { AnalysisResult, ProcessedReference } from '../types';
import { Icons } from './Icons';
import { ScoringConfig, ScoringEngine } from '../services/scoring/ScoringEngine';
import {
  extractBibFileNameFromTex,
  sortBibTexEntriesAlphabetically,
  upsertAndSortBibTexEntries,
  serializeToBibTex,
} from '../services/parsers/BibTexFileEditor';
import { LatexParser } from '../services/parsers/LatexParser';
import { downloadTextFile } from '../utils/downloadTextFile';

interface BibliographyDashboardProps {
  result: AnalysisResult;
  manuscriptText: string;
  bibliographyText: string;
  onUpdate: (newManuscript: string, newBib: string) => void;
  scoringConfig: ScoringConfig;
}

export const BibliographyDashboard: React.FC<BibliographyDashboardProps> = ({
  result,
  manuscriptText,
  bibliographyText,
  onUpdate,
  scoringConfig,
}) => {
  const [status, setStatus] = useState<string>('');

  const computeScore = (scores: any) => {
    return new ScoringEngine(scoringConfig).computeWeightedTotal(scores);
  };

  // Instantiate parser
  const latexParser = useMemo(() => new LatexParser(), []);

  // 1. Identify which references are actually cited in the manuscript
  const citedKeys = useMemo(() => {
    return new Set(latexParser.extractCitations(manuscriptText));
  }, [manuscriptText, latexParser]);

  // 2. Filter references: Must be cited AND have Score > 0
  const validReferences = useMemo(() => {
    return Object.values(result.references)
      .filter(ref => {
        const score = ref.scores ? computeScore(ref.scores) : 0;
        const isCited = citedKeys.has(ref.id);
        // User Requirement: Exclude score 0 and ensure they are cited
        return isCited && score > 0;
      })
      .sort((a, b) => a.id.localeCompare(b.id));
  }, [result.references, citedKeys, scoringConfig]);

  const avgScore = validReferences.length > 0
    ? validReferences.reduce((acc, ref) => acc + (ref.scores ? computeScore(ref.scores) : 0), 0) / validReferences.length
    : 0;

  const bibFilename = useMemo(() => {
    return extractBibFileNameFromTex(manuscriptText) ?? 'references.bib';
  }, [manuscriptText]);

  const handleSortBib = () => {
    // For the "Sort" button, we might still want to sort the *original* full bibliography
    // or maybe just the valid ones? 
    // Usually "Sort" implies organizing the user's input. 
    // But if we are enforcing a filter, maybe we should only keep valid ones?
    // Let's keep the original behavior for the "Sort" button (sorting the raw text) 
    // to avoid data loss during simple editing, but the downloads will be filtered.
    const nextBib = sortBibTexEntriesAlphabetically(bibliographyText);
    onUpdate(manuscriptText, nextBib);
    setStatus('Sorted bibliography alphabetically by citation key.');
  };

  const handleDownloadTex = () => {
    // Identify references that should be removed (Score === 0)
    // We only care about removing citations for references that exist but have 0 score.
    const lowScoreKeys = Object.values(result.references)
        .filter(ref => {
             const score = ref.scores ? computeScore(ref.scores) : 0;
             return score === 0;
        })
        .map(ref => ref.id);

    const cleanTex = latexParser.removeCitations(manuscriptText, lowScoreKeys);
    downloadTextFile('manuscript.tex', cleanTex, 'text/x-tex');
    setStatus('Downloaded cleaned .tex file (removed low-score citations).');
  };

  const handleDownloadBib = () => {
    // Generate BibTeX only for the valid references (Score > 0 && Cited)
    const bibContent = serializeToBibTex(validReferences);
    downloadTextFile(bibFilename, bibContent, 'text/x-bibtex');
    setStatus('Downloaded filtered .bib file (Cited & Score > 0).');
  };

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card p-6 rounded-2xl flex items-center justify-between border border-slate-800 bg-slate-900/50">
          <div>
            <p className="text-slate-400 text-sm font-medium mb-1">Valid References</p>
            <h3 className="text-3xl font-display font-bold text-white">{validReferences.length}</h3>
          </div>
          <div className="p-3 bg-slate-700/50 rounded-xl text-slate-200">
            <Icons.Bibliography className="w-6 h-6" />
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl flex items-center justify-between border border-slate-800 bg-slate-900/50">
          <div>
            <p className="text-slate-400 text-sm font-medium mb-1">Average RefScore</p>
            <h3 className={`text-3xl font-display font-bold ${
                avgScore >= 40 ? 'text-brand-400' : avgScore >= 25 ? 'text-slate-200' : avgScore >= 18 ? 'text-slate-400' : 'text-red-400'
            }`}>
              {avgScore.toFixed(1)}
            </h3>
          </div>
          <div className={`p-3 rounded-xl ${
              avgScore >= 40 ? 'bg-brand-500/10 text-brand-400' : avgScore >= 25 ? 'bg-slate-500/10 text-slate-200' : avgScore >= 18 ? 'bg-slate-700/30 text-slate-400' : 'bg-red-500/10 text-red-400'
          }`}>
            <Icons.Chart className="w-6 h-6" />
          </div>
        </div>

        <div className="glass-card p-6 rounded-2xl flex items-center justify-between border border-slate-800 bg-slate-900/50">
          <div>
            <p className="text-slate-400 text-sm font-medium mb-1">Recent Sources</p>
            <h3 className="text-3xl font-display font-bold text-white">
              {validReferences.filter(r => r.year >= new Date().getFullYear() - 5).length}
            </h3>
          </div>
          <div className="p-3 bg-brand-500/10 rounded-xl text-brand-400">
            <Icons.Recency className="w-6 h-6" />
          </div>
        </div>
      </div>

      {/* Reference List */}
      <div className="glass-card rounded-3xl border border-slate-800 overflow-hidden">
        <div className="p-6 bg-slate-900/80 border-b border-slate-800 flex flex-col gap-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <h3 className="font-display font-bold text-xl text-white flex items-center gap-2">
              <Icons.Bibliography className="w-5 h-5 text-brand-400" />
              Bibliography Tools
            </h3>

            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleSortBib}
                className="px-4 py-2 rounded-xl text-sm font-bold bg-slate-800 text-white hover:bg-slate-700 transition-colors border border-slate-700"
              >
                Sort A→Z
              </button>
              <button
                onClick={handleDownloadTex}
                className="px-4 py-2 rounded-xl text-sm font-bold bg-slate-900 text-slate-200 hover:bg-slate-800 transition-colors border border-slate-700"
              >
                Download .tex
              </button>
              <button
                onClick={handleDownloadBib}
                className="px-4 py-2 rounded-xl text-sm font-bold bg-slate-900 text-slate-200 hover:bg-slate-800 transition-colors border border-slate-700"
              >
                Download .bib
              </button>
            </div>
          </div>

          {status && (
            <div className="text-sm text-slate-400 flex items-center gap-2">
              <Icons.Activity className="w-4 h-4 text-brand-400" />
              {status}
            </div>
          )}
        </div>

        <div className="p-6 bg-slate-900/60 border-b border-slate-800 flex justify-between items-center">
          <h3 className="font-display font-bold text-xl text-white flex items-center gap-2">
            <Icons.Chart className="w-5 h-5 text-brand-400" />
            Bibliography Analysis (Filtered)
          </h3>
        </div>
        
        <div className="divide-y divide-slate-800">
            {validReferences.length === 0 ? (
                <div className="p-8 text-center text-slate-500">
                    No references found that are both cited and have a score &gt; 0.
                </div>
            ) : (
                validReferences.map((ref) => {
                const score = ref.scores ? computeScore(ref.scores) : 0;
                return (
                    <div key={ref.id} className="p-6 hover:bg-slate-800/30 transition-colors group">
                        <div className="flex flex-col md:flex-row gap-6">
                            {/* Score Badge */}
                            <div className="flex-shrink-0">
                                <div className={`w-16 h-16 rounded-2xl flex flex-col items-center justify-center border ${
                                    score >= 70 ? 'bg-brand-500/10 border-brand-500/20 text-brand-400' : 
                                    score >= 50 ? 'bg-slate-500/10 border-slate-500/20 text-slate-200' :
                                    score >= 30 ? 'bg-slate-700/30 border-slate-700/50 text-slate-400' : 
                                    'bg-red-500/10 border-red-500/20 text-red-400'
                                }`}>
                                    <span className="text-xl font-bold">{score.toFixed(0)}</span>
                                    <span className="text-[10px] uppercase font-bold opacity-70">Score</span>
                                </div>
                            </div>

                            {/* Content */}
                            <div className="flex-1 space-y-3">
                                <div className="flex items-start justify-between gap-4">
                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700 font-mono">
                                                @{ref.id}
                                            </span>
                                            <span className="text-xs font-medium text-slate-500">{ref.year}</span>
                                        </div>
                                        <h4 className="font-bold text-slate-200 text-lg leading-snug group-hover:text-brand-300 transition-colors">
                                            {ref.title}
                                        </h4>
                                        <p className="text-sm text-slate-400 mt-1">
                                            {ref.authors.join(', ')} • <span className="italic text-slate-500">{ref.venue || 'Unknown Venue'}</span>
                                        </p>
                                    </div>
                                </div>

                                {/* Abstract Preview */}
                                <div className="relative">
                                    <p className="text-sm text-slate-500 leading-relaxed line-clamp-2 pl-3 border-l-2 border-slate-700">
                                        {ref.abstract}
                                    </p>
                                </div>

                                {/* BibTeX Snippet (Collapsible/Optional) */}
                                <div className="mt-4 pt-4 border-t border-slate-800/50">
                                    <code className="block text-xs font-mono text-slate-500 bg-slate-950/50 p-3 rounded-lg overflow-x-auto border border-slate-800">
                                        {`@article{${ref.id},
  author = {${ref.authors.join(' and ')}},
  title = {${ref.title}},
  year = {${ref.year}},
  journal = {${ref.venue || ''}}
}`}
                                    </code>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            })
        )}
        </div>
      </div>
    </div>
  );
};
