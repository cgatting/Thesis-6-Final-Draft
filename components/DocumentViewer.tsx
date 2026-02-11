import React, { useState } from 'react';
import { AnalysisResult, DimensionScores, ProcessedReference } from '../types';
import { ScoreRadar } from './RadarChart';
import { computeWeightedTotal } from '../services/scoring/ScoringEngine';
import { Icons } from './Icons';
import { CitationFinderService } from '../services/CitationFinderService';

interface DocumentViewerProps {
  result: AnalysisResult;
  onReset: () => void;
  onUpdate: (newManuscript: string, newBib: string) => void;
  manuscriptText: string;
  bibliographyText: string;
}

// Matches \cite{}, \parencite{}, \textcite{} including arguments like [p. 1]
const CITATION_SPLIT_REGEX = /(\\(?:cite|parencite|textcite|footcite)[a-zA-Z]*\*?(?:\[[^\]]*\])*\{[^}]+\})/;

export const DocumentViewer: React.FC<DocumentViewerProps> = ({ result, onReset, onUpdate, manuscriptText, bibliographyText }) => {
  const [findingSource, setFindingSource] = useState(false);
  const [betterSources, setBetterSources] = useState<ProcessedReference[]>([]);
  const [searchPerformed, setSearchPerformed] = useState(false);
  const [updateStatus, setUpdateStatus] = useState<'idle' | 'success'>('idle');
  const [finderService] = useState(() => new CitationFinderService());

  const [activeAnalysis, setActiveAnalysis] = useState<{
    type: 'citation' | 'sentence';
    sentenceIdx: number;
    refId?: string;
    scores?: DimensionScores;
    notes?: string[];
    text?: string;
    category?: 'missing_citation' | 'impact' | 'gap';
    triggerPhrase?: string;
  } | null>(null);

  const missingCitations = result.analyzedSentences
    .map((s, idx) => ({ ...s, idx }))
    .filter(s => s.isMissingCitation);

  const highImpactSentences = result.analyzedSentences
    .map((s, idx) => ({ ...s, idx }))
    .filter(s => s.isHighImpact);

  const identifiedGaps = result.analyzedSentences
    .map((s, idx) => ({ ...s, idx }))
    .filter(s => s.gapIdentified);

  const DIMENSION_DESCRIPTIONS: Record<string, string> = {
    alignment: "Measures how well the reference supports the specific claim it is citing.",
    numbers: "Evaluates the presence and precision of quantitative data in the reference.",
    entities: "Checks for named entity recognition and relevance to the domain.",
    methods: "Assesses the clarity and robustness of the methodology described.",
    recency: "Scores the reference based on its publication date relative to the current year.",
    authority: "Indicators of the venue's or author's impact and reputation."
  };

  const getScoreColor = (score: number) => {
    if (score >= 40) return 'text-green-400 bg-green-500/10 border-green-500/20 hover:bg-green-500/20 ring-green-500/30';
    if (score >= 25) return 'text-blue-400 bg-blue-500/10 border-blue-500/20 hover:bg-blue-500/20 ring-blue-500/30';
    if (score >= 18) return 'text-amber-400 bg-amber-500/10 border-amber-500/20 hover:bg-amber-500/20 ring-amber-500/30';
    return 'text-red-400 bg-red-500/10 border-red-500/20 hover:bg-red-500/20 ring-red-500/30';
  };

  return (
    <div className="flex flex-col lg:flex-row h-[85vh] gap-8 animate-fade-in">
      
      {/* Left: Document View */}
      <div className="flex-1 glass-card rounded-3xl flex flex-col overflow-hidden relative group border border-slate-800">
        <div className="absolute top-0 left-0 right-0 h-1.5 bg-brand-600 z-10"></div>
        
        {/* Header */}
        <div className="p-5 border-b border-slate-800 bg-slate-900 flex justify-between items-center sticky top-0 z-20">
          <h2 className="font-display font-bold text-slate-100 flex items-center gap-3 text-lg">
            <div className="p-2 bg-brand-500/10 rounded-lg text-brand-400">
               <Icons.Manuscript className="w-5 h-5" />
            </div>
            Manuscript Review
          </h2>
          <div className="text-xs font-medium text-slate-400 flex items-center gap-2 bg-slate-800 px-3 py-1.5 rounded-full border border-slate-800 shadow-sm">
            <Icons.Search className="w-3.5 h-3.5 text-brand-400" />
            Click citation pills to inspect
          </div>
        </div>
        
        {/* Content */}
        <div className="flex-1 overflow-y-auto p-8 md:p-12 font-serif leading-loose text-lg text-slate-300 selection:bg-brand-500/30 selection:text-brand-100 scroll-smooth custom-scrollbar">
          <div className="max-w-3xl mx-auto">
            {result.analyzedSentences.map((sent, sIdx) => {
              let highlightClass = 'hover:bg-slate-800/30 border-transparent border-b-2';
              let category: 'missing_citation' | 'impact' | 'gap' | undefined;

              if (sent.isMissingCitation) {
                highlightClass = 'bg-red-500/5 hover:bg-red-500/10 border-red-500/30';
                category = 'missing_citation';
              } else if (sent.gapIdentified) {
                highlightClass = 'bg-amber-500/5 hover:bg-amber-500/10 border-amber-500/30';
                category = 'gap';
              } else if (sent.isHighImpact) {
                highlightClass = 'bg-purple-500/5 hover:bg-purple-500/10 border-purple-500/30';
                category = 'impact';
              }

              const isActive = activeAnalysis?.sentenceIdx === sIdx;
              if (isActive) {
                highlightClass = 'bg-brand-500/10 rounded lg:px-1 shadow-sm ring-1 ring-brand-500/20 border-transparent';
              }

              return (
              <span 
                key={sIdx} 
                className={`transition-all duration-300 decoration-clone box-decoration-clone cursor-pointer px-0.5 rounded ${highlightClass}`}
                onClick={() => {
                  if (category) {
                    setActiveAnalysis({
                      type: 'sentence',
                      sentenceIdx: sIdx,
                      notes: sent.analysisNotes,
                      text: sent.text,
                      category,
                      triggerPhrase: sent.triggerPhrase
                    });
                  } else {
                    setActiveAnalysis(null);
                  }
                }}
              >
                {sent.text.split(CITATION_SPLIT_REGEX).map((part, pIdx) => {
                  // Check if this part is a citation command
                  if (part.startsWith('\\') && (part.includes('cite') || part.includes('parencite'))) {
                     // Extract keys: get content between last { and }
                     const match = /\{([^}]+)\}/.exec(part);
                     if (match) {
                      const keys = match[1].split(',').map(k => k.trim());
                      return (
                        <span key={pIdx} className="inline-flex flex-wrap gap-1.5 mx-1.5 align-baseline">
                          {keys.map((key) => {
                            const scores = sent.scores?.[key];
                            const total = scores ? computeWeightedTotal(scores) : 0;
                            const isActiveRef = activeAnalysis?.type === 'citation' && activeAnalysis.refId === key && activeAnalysis.sentenceIdx === sIdx;
                            
                            return (
                              <button
                                key={key}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (scores) setActiveAnalysis({ 
                                    type: 'citation',
                                    sentenceIdx: sIdx, 
                                    refId: key, 
                                    scores 
                                  });
                                }}
                                className={`
                                  group relative inline-flex items-center justify-center px-3 py-0.5 text-sm font-sans font-bold rounded-full border transition-all duration-200 cursor-pointer select-none
                                  ${scores ? getScoreColor(total) : 'text-slate-400 bg-slate-800 border-slate-700'}
                                  ${isActiveRef ? 'ring-2 ring-offset-2 ring-offset-slate-900 scale-110 shadow-lg z-10' : 'hover:scale-105 hover:shadow-md'}
                                `}
                              >
                                <span className="opacity-50 text-[10px] mr-1 font-normal uppercase tracking-wider">REF</span>
                                {key}
                              </button>
                            );
                          })}
                        </span>
                      );
                     }
                  }
                  return <span key={pIdx}>{part}</span>;
                })}
                {/* Space between sentences */}
                {' '} 
              </span>
            );
            })}
          </div>
        </div>
      </div>

      {/* Right: Analysis Sidebar */}
      <div className="w-full lg:w-[420px] flex flex-col gap-6 shrink-0">
        
        {/* Active Analysis Card */}
        {activeAnalysis ? (
          <div className="glass-card rounded-3xl shadow-xl border border-slate-800 overflow-hidden flex flex-col h-full animate-slide-in-right relative">
             
             {activeAnalysis.type === 'citation' && activeAnalysis.refId && activeAnalysis.scores ? (
               <>
                 {/* Header */}
                 <div className="p-6 bg-slate-900 text-white relative overflow-hidden shrink-0 border-b border-slate-800">
                   
                   <div className="relative z-10">
                     <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/5 text-[10px] font-bold uppercase tracking-widest mb-3 backdrop-blur-md border border-white/10">
                       <Icons.Analyzing className="w-3 h-3 text-brand-300" /> 
                       Citation Analysis
                     </span>
                     <h3 className="font-display font-bold text-2xl truncate tracking-tight" title={activeAnalysis.refId}>
                       [{activeAnalysis.refId}]
                     </h3>
                   </div>
                 </div>
                 
                 <div className="p-6 space-y-6 overflow-y-auto flex-1 custom-scrollbar bg-slate-900/30">
                    {/* Total Score Display */}
                    <div className="flex items-center justify-between p-5 glass-card rounded-2xl hover:border-brand-500/30 transition-all">
                      <div className="flex flex-col">
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Total RefScore</span>
                        <span className="text-xs text-slate-500 font-medium mt-1">Weighted Metric</span>
                      </div>
                      <div className="flex items-baseline gap-1">
                        <span className={`text-4xl font-display font-bold tracking-tighter ${
                          computeWeightedTotal(activeAnalysis.scores) >= 70 ? 'text-green-400' : 
                          computeWeightedTotal(activeAnalysis.scores) >= 50 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                          {computeWeightedTotal(activeAnalysis.scores).toFixed(2)}
                        </span>
                        <span className="text-sm text-slate-500 font-medium">/100</span>
                      </div>
                    </div>

                    {/* Radar Chart */}
                    <div className="glass-card rounded-2xl p-4 relative overflow-hidden bg-slate-800/20">
                       <div className="absolute top-0 left-0 w-full h-1 bg-slate-800"></div>
                       <div className="h-56 w-full flex items-center justify-center">
                          <ScoreRadar data={activeAnalysis.scores} />
                       </div>
                    </div>

                    {/* Reference Details */}
                    <div className="space-y-3">
                       <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                         <Icons.Bibliography className="w-3 h-3 text-brand-400" /> Reference Metadata
                       </h4>
                       <div className="glass-card p-5 rounded-2xl hover:border-brand-500/30 transition-colors group">
                         <p className="font-bold text-slate-200 mb-3 leading-snug group-hover:text-brand-400 transition-colors">
                           {result.references[activeAnalysis.refId]?.title || "Reference Not Found"}
                         </p>
                         <div className="flex flex-wrap gap-2 mb-4">
                            <span className="px-2.5 py-1 bg-slate-800 border border-white/5 rounded-md text-xs text-slate-400 font-bold">
                               {result.references[activeAnalysis.refId]?.year || "N/A"}
                            </span>
                            <span className="px-2.5 py-1 bg-slate-800 border border-white/5 rounded-md text-xs text-slate-400 font-medium truncate max-w-[200px]">
                               {result.references[activeAnalysis.refId]?.authors[0] || "Unknown Author"} et al.
                            </span>
                         </div>
                         <p className="text-slate-500 text-xs leading-relaxed border-t border-white/5 pt-3 italic">
                           {result.references[activeAnalysis.refId]?.abstract 
                             ? (result.references[activeAnalysis.refId].abstract.length > 120 
                                 ? result.references[activeAnalysis.refId].abstract.substring(0, 120) + "..." 
                                 : result.references[activeAnalysis.refId].abstract)
                             : "No abstract available."}
                         </p>
                      </div>

                      {/* Find Better Source Button */}
                      <button
                        onClick={async () => {
                            if (!activeAnalysis.refId) return;
                            setFindingSource(true);
                            setBetterSources([]);
                            setSearchPerformed(false);
                            setUpdateStatus('idle');
                            
                            const currentRef = result.references[activeAnalysis.refId];
                            const contextSentence = result.analyzedSentences[activeAnalysis.sentenceIdx]?.text || "";
                            
                            const better = await finderService.findBetterSources(currentRef, contextSentence);
                            setBetterSources(better);
                            setSearchPerformed(true);
                            setFindingSource(false);
                        }}
                        disabled={findingSource}
                        className="w-full py-3 bg-brand-600/20 hover:bg-brand-600/30 border border-brand-500/30 text-brand-300 hover:text-brand-200 text-xs font-bold rounded-xl transition-all flex items-center justify-center gap-2 group"
                      >
                        {findingSource ? (
                            <>
                                <Icons.Analyzing className="w-3.5 h-3.5 animate-spin" />
                                Analyzing Sources...
                            </>
                        ) : (
                            <>
                                <Icons.Search className="w-3.5 h-3.5 group-hover:scale-110 transition-transform" />
                                Find Better Sources
                            </>
                        )}
                      </button>

                      {/* Better Source Results */}
                      {searchPerformed && betterSources.length === 0 && (
                          <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-xl animate-scale-in">
                              <div className="flex items-center gap-2 text-slate-300">
                                  <Icons.CheckCircle className="w-4 h-4 text-green-400" />
                                  <p className="text-sm font-bold">The best option is currently in the document already.</p>
                              </div>
                          </div>
                      )}

                      {betterSources.length > 0 && (
                          <div className="space-y-3 animate-scale-in">
                              <div className="flex justify-between items-center">
                                  <span className="text-[10px] font-bold text-green-400 uppercase tracking-wider flex items-center gap-1">
                                    <Icons.Success className="w-3 h-3" /> Recommended Sources
                                  </span>
                              </div>
                              
                              {betterSources.map((source, idx) => {
                                const score = source.scores ? computeWeightedTotal(source.scores) : 0;
                                return (
                                <div key={source.id || idx} className="p-4 bg-green-500/10 border border-green-500/20 rounded-xl">
                                    <div className="flex justify-between items-start mb-3">
                                        <div className="flex items-center gap-2">
                                            <span className="text-[10px] font-bold text-green-300 bg-green-500/20 px-2 py-0.5 rounded-full border border-green-500/20">Option {idx + 1}</span>
                                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                                                score >= 70 ? 'text-green-300 bg-green-500/20 border-green-500/20' :
                                                score >= 50 ? 'text-amber-300 bg-amber-500/20 border-amber-500/20' :
                                                'text-red-300 bg-red-500/20 border-red-500/20'
                                            }`}>
                                                {score.toFixed(0)}/100 Match
                                            </span>
                                        </div>
                                    </div>
                                    <p className="text-sm font-bold text-slate-200 mb-1 leading-snug">{source.title}</p>
                                    <p className="text-xs text-slate-400 mb-4">{source.authors[0]} et al. ({source.year}) • {source.venue}</p>
                                    
                                    <button
                                        onClick={() => {
                                            if (!activeAnalysis.refId) return;
                                            
                                            const { manuscript, bib } = finderService.updateFiles(
                                                activeAnalysis.refId,
                                                source,
                                                manuscriptText,
                                                bibliographyText
                                            );
                                            
                                            setUpdateStatus('success');
                                            // Trigger parent update after a brief delay to show success state
                                            setTimeout(() => {
                                                onUpdate(manuscript, bib);
                                                setUpdateStatus('idle');
                                                setBetterSources([]);
                                                setSearchPerformed(false);
                                                setActiveAnalysis(null);
                                            }, 1500);
                                        }}
                                        disabled={updateStatus === 'success'}
                                        className={`w-full py-2 text-xs font-bold rounded-lg transition-all flex items-center justify-center gap-2 ${
                                          updateStatus === 'success' 
                                            ? 'bg-green-600 text-white cursor-default' 
                                            : 'bg-green-600 hover:bg-green-500 text-white shadow-lg shadow-green-900/20'
                                        }`}
                                    >
                                        {updateStatus === 'success' ? (
                                          <>
                                            <Icons.CheckCircle className="w-3.5 h-3.5" /> Files Updated Successfully
                                          </>
                                        ) : (
                                          <>
                                            Select & Replace Citation
                                          </>
                                        )}
                                    </button>
                                </div>
                              );
                              })}
                          </div>
                      )}
                    </div>

                    {/* Score Breakdown (Mini Table) */}
                     <div className="space-y-3">
                       <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                         <Icons.Chart className="w-3 h-3 text-brand-400" /> Dimension Breakdown
                       </h4>
                       <div className="grid grid-cols-2 gap-3">
                          {Object.entries(activeAnalysis.scores).map(([key, val]) => (
                            <div 
                              key={key} 
                              title={DIMENSION_DESCRIPTIONS[key.toLowerCase()] || "Analysis metric"}
                              className="flex justify-between items-center p-3 glass-card rounded-xl text-xs hover:border-brand-500/30 transition-colors cursor-help"
                            >
                               <span className="capitalize text-slate-400 font-bold">{key}</span>
                               <span className={`font-bold px-2 py-0.5 rounded ${
                                 val > 70 ? 'bg-green-500/10 text-green-400' : 
                                 val > 50 ? 'bg-amber-500/10 text-amber-400' : 'bg-red-500/10 text-red-400'
                               }`}>{val.toFixed(2)}</span>
                            </div>
                          ))}
                       </div>
                     </div>
                 </div>
               </>
             ) : (
               <>
                 <div className="p-6 bg-slate-900 text-white relative overflow-hidden shrink-0 border-b border-slate-800">
                    <div className="relative z-10">
                        {activeAnalysis.category === 'missing_citation' && (
                             <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/10 text-red-400 text-[10px] font-bold uppercase tracking-widest mb-3 border border-red-500/20">
                               <Icons.Warning className="w-3 h-3" /> Missing Citation
                             </span>
                        )}
                        {activeAnalysis.category === 'impact' && (
                             <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-purple-500/10 text-purple-400 text-[10px] font-bold uppercase tracking-widest mb-3 border border-purple-500/20">
                               <Icons.Authority className="w-3 h-3" /> High Impact
                             </span>
                        )}
                        {activeAnalysis.category === 'gap' && (
                             <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-400 text-[10px] font-bold uppercase tracking-widest mb-3 border border-amber-500/20">
                               <Icons.Search className="w-3 h-3" /> Research Gap
                             </span>
                        )}
                        <h3 className="font-display font-bold text-xl leading-tight">
                            Sentence Analysis
                        </h3>
                    </div>
                 </div>
                 
                 <div className="p-6 space-y-6 overflow-y-auto flex-1 custom-scrollbar bg-slate-900/30">
                    <div className="glass-card p-5 rounded-2xl bg-slate-800/50 border border-white/5">
                        <p className="text-slate-300 italic text-lg leading-relaxed">
                            "{activeAnalysis.text}"
                        </p>
                    </div>
                    
                    <div className="space-y-4">
                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                            <Icons.Analyzing className="w-3 h-3 text-brand-400" /> Analysis Insights
                        </h4>
                        
                        {/* Trigger Phrase Display */}
                        {activeAnalysis.triggerPhrase && (
                           <div className="p-3 bg-brand-500/10 border border-brand-500/20 rounded-xl mb-2">
                              <span className="text-[10px] text-brand-300 font-bold uppercase tracking-wider block mb-1">
                                Detected Trigger
                              </span>
                              <p className="text-sm font-bold text-brand-100">
                                "{activeAnalysis.triggerPhrase}"
                              </p>
                           </div>
                        )}

                        {activeAnalysis.notes?.map((note, idx) => (
                            <div key={idx} className="flex gap-3 p-4 glass-card rounded-xl border border-slate-700/50">
                                <div className="mt-0.5">
                                    <Icons.Info className="w-4 h-4 text-brand-400" />
                                </div>
                                <p className="text-sm text-slate-300 font-medium">{note}</p>
                            </div>
                        ))}
                    </div>
                 </div>
               </>
             )}

          </div>
        ) : (
          <div className="glass-card rounded-3xl overflow-hidden flex flex-col h-full border border-slate-800 relative">
             <div className="p-6 bg-slate-900 border-b border-slate-800">
               <h3 className="font-display font-bold text-xl text-white flex items-center gap-2">
                 <Icons.Analyzing className="w-5 h-5 text-brand-400" />
                 Analysis Overview
               </h3>
               <p className="text-sm text-slate-400 mt-1">
                 Select a category below to jump to findings.
               </p>
             </div>

             <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-8">
               
               {/* Missing Citations Section */}
               <div>
                 <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                   <Icons.Warning className="w-3 h-3" /> Missing Citations ({missingCitations.length})
                 </h4>
                 {missingCitations.length > 0 ? (
                   <div className="space-y-2">
                     {missingCitations.map((item, i) => (
                       <button 
                         key={i}
                         onClick={() => {
                            const element = document.querySelectorAll('.max-w-3xl span')[item.idx];
                            element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            setActiveAnalysis({
                              type: 'sentence',
                              sentenceIdx: item.idx,
                              notes: item.analysisNotes,
                              text: item.text,
                              category: 'missing_citation',
                              triggerPhrase: item.triggerPhrase
                            });
                         }}
                         className="w-full text-left p-3 rounded-xl bg-red-500/5 border border-red-500/10 hover:bg-red-500/10 transition-colors group"
                       >
                         <div className="flex items-start gap-2">
                           <span className="mt-1 w-1.5 h-1.5 rounded-full bg-red-500 shrink-0" />
                           <p className="text-xs text-slate-300 line-clamp-2 group-hover:text-white transition-colors">
                             "{item.text}"
                           </p>
                         </div>
                       </button>
                     ))}
                   </div>
                 ) : (
                   <p className="text-xs text-slate-500 italic px-2">No missing citations detected.</p>
                 )}
               </div>

               {/* High Impact Section */}
               <div>
                 <h4 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                   <Icons.Authority className="w-3 h-3" /> High Impact ({highImpactSentences.length})
                 </h4>
                 {highImpactSentences.length > 0 ? (
                   <div className="space-y-2">
                     {highImpactSentences.map((item, i) => (
                       <button 
                         key={i}
                         onClick={() => {
                            const element = document.querySelectorAll('.max-w-3xl span')[item.idx];
                            element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            setActiveAnalysis({
                              type: 'sentence',
                              sentenceIdx: item.idx,
                              notes: item.analysisNotes,
                              text: item.text,
                              category: 'impact',
                              triggerPhrase: item.triggerPhrase
                            });
                         }}
                         className="w-full text-left p-3 rounded-xl bg-purple-500/5 border border-purple-500/10 hover:bg-purple-500/10 transition-colors group"
                       >
                         <div className="flex items-start gap-2">
                           <span className="mt-1 w-1.5 h-1.5 rounded-full bg-purple-500 shrink-0" />
                           <p className="text-xs text-slate-300 line-clamp-2 group-hover:text-white transition-colors">
                             "{item.text}"
                           </p>
                         </div>
                       </button>
                     ))}
                   </div>
                 ) : (
                   <p className="text-xs text-slate-500 italic px-2">No high impact sentences detected.</p>
                 )}
               </div>

               {/* Gaps Section */}
               <div>
                 <h4 className="text-xs font-bold text-amber-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                   <Icons.Search className="w-3 h-3" /> Research Gaps ({identifiedGaps.length})
                 </h4>
                 {identifiedGaps.length > 0 ? (
                   <div className="space-y-2">
                     {identifiedGaps.map((item, i) => (
                       <button 
                         key={i}
                         onClick={() => {
                            const element = document.querySelectorAll('.max-w-3xl span')[item.idx];
                            element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                            setActiveAnalysis({
                              type: 'sentence',
                              sentenceIdx: item.idx,
                              notes: item.analysisNotes,
                              text: item.text,
                              category: 'gap',
                              triggerPhrase: item.triggerPhrase
                            });
                         }}
                         className="w-full text-left p-3 rounded-xl bg-amber-500/5 border border-amber-500/10 hover:bg-amber-500/10 transition-colors group"
                       >
                         <div className="flex items-start gap-2">
                           <span className="mt-1 w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0" />
                           <p className="text-xs text-slate-300 line-clamp-2 group-hover:text-white transition-colors">
                             "{item.text}"
                           </p>
                         </div>
                       </button>
                     ))}
                   </div>
                 ) : (
                  <div 
                    onClick={() => {
                      setActiveAnalysis({
                        type: 'sentence',
                        sentenceIdx: -1,
                        notes: [
                          "The system identifies research gaps by looking for specific markers:",
                          "• Explicit statements: 'gap in the literature', 'remains to be', 'future work'",
                          "• Uncertainty markers: 'unknown', 'unclear', 'little is known', 'not well understood'",
                          "• Limitation markers: 'limited', 'lack of', 'needs further', 'unresolved'",
                          "Recommendation: Explicitly stating gaps helps position your work's contribution clearly."
                        ],
                        text: "Research Gap Detection Logic",
                        category: 'gap',
                        triggerPhrase: "How it works"
                      });
                    }}
                    className="flex gap-3 p-4 rounded-xl bg-amber-500/5 border border-amber-500/10 hover:bg-amber-500/10 transition-colors cursor-pointer group"
                  >
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold mt-0.5 border border-amber-500/20 group-hover:scale-110 transition-transform">?</span>
                    <p className="text-sm text-slate-300 leading-snug group-hover:text-white transition-colors">
                      No major gaps detected. <span className="text-amber-400/80 group-hover:text-amber-400">Click to see how to highlight research gaps.</span>
                    </p>
                  </div>
                )}
               </div>

             </div>
          </div>
        )}
      </div>
    </div>
  );
};