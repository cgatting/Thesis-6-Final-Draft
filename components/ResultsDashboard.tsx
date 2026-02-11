import React, { useState } from 'react';
import { AnalysisResult, ProcessedReference } from '../types';
import { ScoreRadar } from './RadarChart';
import { Icons } from './Icons';
import { computeWeightedTotal } from '../services/scoring/ScoringEngine';

interface ResultsDashboardProps {
  result: AnalysisResult;
  onReset: () => void;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ result, onReset }) => {
  const [filter, setFilter] = useState('');
  const [sortField, setSortField] = useState<'id' | 'title' | 'year' | 'score'>('id');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const references = Object.values(result.references) as ProcessedReference[];
  
  // Filter out references that are not cited in the manuscript
  const citedKeys = new Set(result.analyzedSentences.flatMap(s => s.citations || []));
  const validReferences = references.filter(ref => citedKeys.has(ref.id));

  const filteredRefs = validReferences.filter(ref => 
    ref.title.toLowerCase().includes(filter.toLowerCase()) || 
    ref.id.toLowerCase().includes(filter.toLowerCase()) ||
    ref.authors.some(author => author.toLowerCase().includes(filter.toLowerCase()))
  );

  const sortedRefs = [...filteredRefs].sort((a, b) => {
    const multiplier = sortDirection === 'asc' ? 1 : -1;
    
    switch (sortField) {
      case 'id':
        return multiplier * a.id.localeCompare(b.id);
      case 'title':
        return multiplier * a.title.localeCompare(b.title);
      case 'year':
        return multiplier * ((a.year || 0) - (b.year || 0));
      case 'score':
        const scoreA = a.scores ? computeWeightedTotal(a.scores) : 0;
        const scoreB = b.scores ? computeWeightedTotal(b.scores) : 0;
        return multiplier * (scoreA - scoreB);
      default:
        return 0;
    }
  });

  const handleSort = (field: 'id' | 'title' | 'year' | 'score') => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const SortIcon = ({ field }: { field: 'id' | 'title' | 'year' | 'score' }) => {
    return (
      <span className="ml-1 inline-flex flex-col h-3 justify-center">
        <Icons.ChevronUp className={`w-2 h-2 -mb-0.5 ${sortField === field && sortDirection === 'asc' ? 'text-brand-400' : 'text-slate-700'}`} />
        <Icons.ChevronDown className={`w-2 h-2 ${sortField === field && sortDirection === 'desc' ? 'text-brand-400' : 'text-slate-700'}`} />
      </span>
    );
  };

  const DIMENSION_DESCRIPTIONS: Record<string, string> = {
    alignment: "Measures how well the reference supports the specific claim it is citing.",
    numbers: "Evaluates the presence and precision of quantitative data in the reference.",
    entities: "Checks for named entity recognition and relevance to the domain.",
    methods: "Assesses the clarity and robustness of the methodology described.",
    recency: "Scores the reference based on its publication date relative to the current year.",
    authority: "Indicators of the venue's or author's impact and reputation."
  };

  return (
    <div className="space-y-8 pb-12 animate-fade-in">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row gap-6 items-start md:items-center justify-between">
        <div>
          <h2 className="text-3xl font-display font-bold text-slate-100 tracking-tight">
            Analysis Report{result.documentTitle ? `: ${result.documentTitle}` : ''}
          </h2>
          <p className="text-slate-400 mt-1 flex items-center gap-2 text-sm font-medium">
            <Icons.Calendar className="w-4 h-4 text-brand-400" />
            Generated on {new Date().toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
          </p>
        </div>
        <div className="flex items-center gap-3 w-full md:w-auto">
          <button className="px-5 py-2.5 text-sm font-bold text-slate-900 bg-slate-100 hover:bg-brand-50 rounded-xl shadow-lg shadow-black/10 hover:shadow-brand-500/20 transition-all flex items-center gap-2 active:scale-95">
            <Icons.Upload className="w-4 h-4 rotate-180" />
            Export
          </button>
        </div>
      </div>

      {/* Overview Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Main Score */}
        <div className="col-span-1 md:col-span-2 bg-slate-900 rounded-3xl p-8 text-slate-100 shadow-xl border border-slate-800">
          
          <div className="relative z-10 flex justify-between items-end h-full">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <p className="text-brand-200 text-xs font-bold uppercase tracking-widest">RefScore Index</p>
              </div>
              <div className="text-6xl md:text-7xl font-display font-bold tracking-tighter leading-none mb-1">
                {result.overallScore}
                <span className="text-3xl text-slate-400 font-normal ml-1">/100</span>
              </div>
              <div className="mt-6 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-100/10 text-sm font-semibold backdrop-blur-md border border-slate-100/10 shadow-lg">
                {result.overallScore >= 40 ? (
                  <><Icons.Success className="w-4 h-4 text-green-400" /> Excellent Quality</>
                ) : result.overallScore >= 25 ? (
                  <><Icons.Warning className="w-4 h-4 text-blue-400" /> Good Quality</>
                ) : result.overallScore >= 18 ? (
                  <><Icons.Warning className="w-4 h-4 text-amber-400" /> Moderate Quality</>
                ) : (
                  <><Icons.Warning className="w-4 h-4 text-red-400" /> Needs Improvement</>
                )}
              </div>
            </div>
            <div className="w-32 h-32 md:w-40 md:h-40 opacity-90 drop-shadow-2xl">
               <ScoreRadar data={result.dimensionScores} />
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="glass-card rounded-3xl p-6 flex flex-col justify-between hover:border-brand-500/30 group">
          <div className="flex items-start justify-between">
            <div className="p-3 bg-blue-500/10 text-blue-400 rounded-2xl group-hover:scale-110 transition-transform duration-300 border border-blue-500/20">
              <Icons.Bibliography className="w-6 h-6" />
            </div>
            {validReferences.length > 20 && (
              <span className="text-[10px] font-bold text-green-400 bg-green-500/10 px-2 py-1 rounded-full border border-green-500/20 uppercase tracking-wide">High Volume</span>
            )}
          </div>
          <div>
            <div className="text-4xl font-display font-bold text-slate-100 mt-4">{validReferences.length}</div>
            <div className="text-slate-400 text-sm font-medium">Total Citations</div>
          </div>
        </div>

        <div className="glass-card rounded-3xl p-6 flex flex-col justify-between hover:border-amber-500/30 group">
          <div className="flex items-start justify-between">
            <div className="p-3 bg-amber-500/10 text-amber-400 rounded-2xl group-hover:scale-110 transition-transform duration-300 border border-amber-500/20">
              <Icons.Clock className="w-6 h-6" />
            </div>
            <span className="text-[10px] font-bold text-amber-400 bg-amber-500/10 px-2 py-1 rounded-full border border-amber-500/20 uppercase tracking-wide">Avg Age</span>
          </div>
          <div>
            <div className="text-4xl font-display font-bold text-white mt-4">
              {validReferences.length > 0 ? Math.round(validReferences.reduce((acc, curr) => acc + (curr.year ? new Date().getFullYear() - curr.year : 0), 0) / validReferences.length) : 0}y
            </div>
            <div className="text-slate-400 text-sm font-medium">Average Reference Age</div>
          </div>
        </div>
      </div>

      {/* Dimension Analysis */}
      <div>
        <h3 className="text-xl font-display font-bold text-white mb-6 flex items-center gap-2">
          <Icons.Layers className="w-5 h-5 text-brand-400" />
          Dimension Breakdown
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {(Object.entries(result.dimensionScores) as [string, number][]).map(([dim, score]) => (
            <div key={dim} className="glass-card rounded-2xl p-5 hover:border-brand-500/30 group relative transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">
              <div className="flex justify-between items-start mb-4">
                <h4 className="font-bold text-slate-200 capitalize flex items-center gap-2">
                  {dim}
                  <Icons.Info className="w-3.5 h-3.5 text-slate-600 group-hover:text-brand-400 transition-colors" />
                </h4>
                <div className={`px-2.5 py-1 rounded-lg text-xs font-bold border ${
                  score >= 40 ? 'bg-green-500/10 text-green-400 border-green-500/20' : 
                  score >= 25 ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' : 
                  score >= 18 ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                  'bg-red-500/10 text-red-400 border-red-500/20'
                }`}>
                  {score.toFixed(2)}/100
                </div>
              </div>
              <div className="w-full bg-slate-700/50 rounded-full h-2 mb-2 overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all duration-1000 ${
                    score >= 40 ? 'bg-green-500' : 
                    score >= 25 ? 'bg-blue-500' : 
                    score >= 18 ? 'bg-amber-500' :
                    'bg-red-500'
                  }`}
                  style={{ width: `${score}%` }}
                ></div>
              </div>
              
              <div className="grid grid-rows-[0fr] group-hover:grid-rows-[1fr] transition-[grid-template-rows] duration-500 ease-in-out">
                <div className="overflow-hidden">
                  <p className="text-xs text-slate-400 mt-4 pt-3 border-t border-white/5 leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity duration-500 delay-100">
                    {DIMENSION_DESCRIPTIONS[dim.toLowerCase()] || "Analysis metric."}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Detailed Reference Table */}
        <div className="lg:col-span-2 glass-card rounded-3xl flex flex-col overflow-hidden border border-slate-800">
          <div className="p-6 border-b border-slate-800 flex flex-col md:flex-row justify-between items-center gap-4 bg-slate-900">
            <h3 className="font-display font-bold text-white flex items-center gap-2 text-lg">
              <Icons.Layers className="w-5 h-5 text-brand-400" />
              Reference Analysis
            </h3>
            <div className="flex items-center gap-3 w-full md:w-auto">
              <div className="relative flex-1 md:w-64">
                <Icons.Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input 
                  type="text" 
                  placeholder="Filter by title, author, or ID..." 
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 text-sm bg-slate-800 border border-slate-700 text-slate-100 rounded-xl focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all shadow-sm placeholder:text-slate-600"
                />
              </div>
              <span className="text-xs font-bold text-slate-400 bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700 whitespace-nowrap">
                {filteredRefs.length} References
              </span>
            </div>
          </div>
          
          <div className="overflow-x-auto custom-scrollbar max-h-[600px]">
            <table className="w-full text-left text-sm text-slate-400">
              <thead className="bg-slate-900 text-xs uppercase font-bold text-slate-500 border-b border-slate-800 sticky top-0 z-10">
                <tr>
                  <th className="px-6 py-4 w-20 tracking-wider cursor-pointer hover:bg-slate-800/50 transition-colors select-none" onClick={() => handleSort('id')}>
                    <div className="flex items-center">ID <SortIcon field="id" /></div>
                  </th>
                  <th className="px-6 py-4 tracking-wider cursor-pointer hover:bg-slate-800/50 transition-colors select-none" onClick={() => handleSort('title')}>
                    <div className="flex items-center">Title / Author <SortIcon field="title" /></div>
                  </th>
                  <th className="px-6 py-4 w-24 tracking-wider cursor-pointer hover:bg-slate-800/50 transition-colors select-none" onClick={() => handleSort('year')}>
                    <div className="flex items-center">Year <SortIcon field="year" /></div>
                  </th>
                  <th className="px-6 py-4 text-center tracking-wider cursor-pointer hover:bg-slate-800/50 transition-colors select-none" onClick={() => handleSort('score')}>
                    <div className="flex items-center justify-center">Score <SortIcon field="score" /></div>
                  </th>
                  <th className="px-6 py-4 w-32 tracking-wider cursor-pointer hover:bg-slate-800/50 transition-colors select-none" onClick={() => handleSort('score')}>
                    <div className="flex items-center">Status <SortIcon field="score" /></div>
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {sortedRefs.map((ref) => {
                  const avgScore = ref.scores 
                    ? computeWeightedTotal(ref.scores) 
                    : 0;
                    
                  return (
                    <tr key={ref.id} className="hover:bg-slate-800/30 transition-colors group">
                      <td className="px-6 py-4 font-mono text-xs text-slate-500 font-bold group-hover:text-brand-400 transition-colors">
                        {ref.id}
                      </td>
                      <td className="px-6 py-4">
                        <div className="font-bold text-slate-200 mb-0.5 line-clamp-1 group-hover:text-brand-400 transition-colors" title={ref.title}>{ref.title}</div>
                        <div className="text-xs text-slate-500 line-clamp-1">{ref.authors.join(', ')}</div>
                      </td>
                      <td className="px-6 py-4 text-slate-500 font-mono text-xs font-medium">
                        {ref.year}
                      </td>
                      <td className="px-6 py-4 text-center">
                        <div className={`inline-flex items-center justify-center font-bold text-sm ${
                           avgScore >= 40 ? 'text-green-400' :
                           avgScore >= 25 ? 'text-blue-400' :
                           avgScore >= 18 ? 'text-amber-400' : 'text-red-400'
                        }`}>
                          {avgScore.toFixed(2)}
                        </div>
                      </td>
                      <td className="px-6 py-4">
                         <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wide border ${
                           avgScore >= 40 ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                           avgScore >= 25 ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' :
                           avgScore >= 18 ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                           'bg-red-500/10 text-red-400 border-red-500/20'
                         }`}>
                           {avgScore >= 40 ? 'Good' : avgScore >= 25 ? 'OK' : avgScore >= 18 ? 'Medium' : 'Bad'}
                         </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {filteredRefs.length === 0 && (
             <div className="p-16 text-center text-slate-500 flex flex-col items-center">
               <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-4">
                  <Icons.Search className="w-6 h-6 text-slate-600" />
               </div>
               <p className="font-medium">No references found matching your filter.</p>
             </div>
          )}
        </div>

        {/* Right Column: Insights & Summary */}
        <div className="space-y-6">
          
          {/* Executive Summary */}
          <div className="glass-card rounded-3xl p-6 relative overflow-hidden group">
            <h3 className="text-white font-bold mb-3 flex items-center gap-2 relative z-10">
              <Icons.Analyzing className="w-5 h-5 text-brand-400" />
              Executive Summary
            </h3>
            <p className="text-sm text-slate-300 leading-relaxed opacity-90 relative z-10 font-medium">
              {result.summary}
            </p>
          </div>

          {/* Critical Gaps */}
          <div className="glass-card rounded-3xl p-6 border border-white/5">
            <h3 className="font-bold text-white mb-4 flex items-center gap-2">
              <Icons.Warning className="w-5 h-5 text-amber-400" />
              Detected Gaps
            </h3>
            
            {result.gaps.length === 0 ? (
               <div className="p-4 bg-green-500/10 rounded-xl border border-green-500/20 text-center">
                 <p className="text-sm text-green-400 font-bold">No critical argumentation gaps detected.</p>
               </div>
             ) : (
               <div className="space-y-3">
                 {result.gaps.map((gap, i) => (
                   <div key={i} className="flex gap-3 p-4 rounded-xl bg-amber-500/5 border border-amber-500/10 hover:bg-amber-500/10 transition-colors cursor-default">
                     <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[10px] font-bold mt-0.5 border border-amber-500/20">
                       {i + 1}
                     </span>
                     <p className="text-sm text-slate-300 leading-snug">{gap}</p>
                   </div>
                 ))}
               </div>
             )}
          </div>
          
           {/* Recommendations */}
           <div className="glass-card rounded-3xl p-6 border border-white/5">
            <h3 className="font-bold text-white mb-4 flex items-center gap-2">
              <Icons.Activity className="w-5 h-5 text-brand-400" />
              Next Steps
            </h3>
            <ul className="space-y-3">
              {[
                `Review references flagged as "Weak" for potential replacement.`,
                `Address the ${result.gaps.length} detected argumentation gaps.`,
                `Ensure high-impact claims have recent citations (last 5 years).`
              ].map((step, i) => (
                <li key={i} className="flex items-start gap-3 text-sm text-slate-300 p-2 hover:bg-slate-800/50 rounded-lg transition-colors">
                  <div className="mt-0.5 p-1 bg-brand-500/20 rounded-full text-brand-400">
                     <Icons.CheckCircle className="w-3 h-3" />
                  </div>
                  <span className="font-medium">{step}</span>
                </li>
              ))}
            </ul>
          </div>

        </div>

      </div>
    </div>
  );
};
