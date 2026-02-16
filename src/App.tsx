import React, { useState, useEffect } from 'react';
import { Icons } from './components/Icons';
import { FileUpload } from './components/FileUpload';
import { DocumentViewer } from './components/DocumentViewer';
import { ResultsDashboard } from './components/ResultsDashboard';
import { BibliographyDashboard } from './components/BibliographyDashboard';
import { LandingPage } from './components/LandingPage';
import { SettingsModal } from './components/SettingsModal';
import { AnalysisService } from './services/AnalysisService';
import { DeepSearchClient } from './services/DeepSearchClient';
import { ScoringConfig, ScoringEngine } from './services/scoring/ScoringEngine';
import { AppState, AnalysisResult } from './types';

function App() {
  const [showLanding, setShowLanding] = useState(true);
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'document' | 'bibliography'>('dashboard');
  const [statusMessage, setStatusMessage] = useState('');
  
  const [manuscriptText, setManuscriptText] = useState('');
  const [bibliographyText, setBibliographyText] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [scrolled, setScrolled] = useState(false);
  const [deepSearchProgress, setDeepSearchProgress] = useState(0);
  
  const [scoringConfig, setScoringConfig] = useState<ScoringConfig>(ScoringEngine.DEFAULT_CONFIG);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [autoGenerateBib, setAutoGenerateBib] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    setScoringConfig(prev => ({ ...prev, deepSearchEnabled: autoGenerateBib }));
  }, [autoGenerateBib]);

  const handleAnalysis = async (configOverride?: ScoringConfig) => {
    const deepSearchFlag = !!autoGenerateBib;

    if (!manuscriptText.trim() || (!autoGenerateBib && !bibliographyText.trim())) {
      setErrorMsg(autoGenerateBib ? "Please provide manuscript content." : "Please provide both manuscript content and a bibliography, or enable Generate References.");
      return;
    }

    setAppState(AppState.PARSING);
    setErrorMsg(null);
    setStatusMessage("Initializing analysis pipeline...");

    try {
      setTimeout(async () => {
          try {
              let nextManuscript = manuscriptText;
              let nextBibliography = bibliographyText;

              if (deepSearchFlag) {
                setStatusMessage("Initializing DeepSearch refinement...");
                const client = new DeepSearchClient();
                const refined = await client.refineDocument(manuscriptText, (pct, msg) => {
                    setDeepSearchProgress(pct);
                    setStatusMessage(msg);
                });
                nextManuscript = refined.processedText;
                nextBibliography = (refined.bibtex && refined.bibtex.trim().length > 0)
                  ? refined.bibtex
                  : (refined.bibliographyText || nextBibliography);
                setManuscriptText(nextManuscript);
                setBibliographyText(nextBibliography);
                setDeepSearchProgress(0); // Reset for next phase
              }

              setStatusMessage("Parsing and Vectorizing content...");
              const service = new AnalysisService(scoringConfig);
              const analysisResult = await service.analyze(nextManuscript, nextBibliography);
              
              setResult(analysisResult);
              setAppState(AppState.RESULTS);
              setActiveTab('dashboard');
          } catch (err: any) {
              console.error(err);
              setAppState(AppState.ERROR);
              setErrorMsg(err.message || "Pipeline failed.");
          }
      }, 800); // Slight delay for UX smoothness

    } catch (err: any) {
      console.error(err);
      setAppState(AppState.ERROR);
      setErrorMsg(err.message || "Pipeline failed.");
    }
  };

  const handleReset = () => {
    setAppState(AppState.IDLE);
    setResult(null);
    setErrorMsg(null);
    setManuscriptText('');
    setBibliographyText('');
    setAutoGenerateBib(false);
  };

  const handleUpdate = async (newManuscript: string, newBib: string) => {
    setManuscriptText(newManuscript);
    setBibliographyText(newBib);
    
    // Trigger re-analysis
    setAppState(AppState.PARSING);
    setStatusMessage("Re-analyzing updated content...");
    
    setTimeout(async () => {
        try {
            const service = new AnalysisService(scoringConfig);
            const analysisResult = await service.analyze(newManuscript, newBib);
            
            setResult(analysisResult);
            setAppState(AppState.RESULTS);
        } catch (err: any) {
            console.error(err);
            setAppState(AppState.ERROR);
            setErrorMsg(err.message || "Update failed.");
        }
    }, 1000);
  };

  if (showLanding) {
    return <LandingPage onStart={() => setShowLanding(false)} />;
  }

  return (
    <div className="min-h-screen font-sans text-slate-100 selection:bg-brand-500/30 selection:text-brand-100 overflow-x-hidden">
      
      {/* Navigation Bar */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrolled ? 'glass py-3' : 'bg-transparent py-5'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3 cursor-pointer group" onClick={handleReset}>
              <div className="bg-brand-600 p-2.5 rounded-xl shadow-lg shadow-brand-500/20 group-hover:scale-105 transition-transform duration-300">
                <Icons.Layers className="w-5 h-5 text-slate-100" />
              </div>
              <span className="font-display font-bold text-xl tracking-tight text-slate-100 group-hover:text-brand-400 transition-colors">
                RefScore
              </span>
            </div>
            
            {appState === AppState.RESULTS && (
               <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1 bg-slate-900/60 backdrop-blur-sm p-1.5 rounded-xl border border-white/10 shadow-inner">
                 <button 
                   onClick={() => setActiveTab('dashboard')}
                   className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-200 ${
                     activeTab === 'dashboard' 
                       ? 'bg-slate-700 text-white shadow-sm ring-1 ring-white/10' 
                       : 'text-slate-400 hover:text-white hover:bg-white/5'
                   }`}
                 >
                   Overview
                 </button>
                 <button 
                   onClick={() => setActiveTab('document')}
                   className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-200 ${
                     activeTab === 'document' 
                       ? 'bg-slate-700 text-white shadow-sm ring-1 ring-white/10' 
                       : 'text-slate-500 hover:text-slate-900 hover:bg-white/50'
                   }`}
                 >
                   Review Manuscript
                 </button>
                 <button 
                   onClick={() => setActiveTab('bibliography')}
                   className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-200 ${
                     activeTab === 'bibliography' 
                       ? 'bg-slate-700 text-white shadow-sm ring-1 ring-white/10' 
                       : 'text-slate-500 hover:text-slate-900 hover:bg-white/50'
                   }`}
                 >
                   Bibliography
                 </button>
               </div>
            )}

            <button 
              onClick={() => setIsSettingsOpen(true)}
              className="p-2.5 text-slate-400 hover:text-white hover:bg-slate-800/50 rounded-xl transition-all border border-transparent hover:border-slate-700"
              title="Configure Scoring"
            >
              <Icons.Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </nav>

      <SettingsModal 
         isOpen={isSettingsOpen}
         onClose={() => setIsSettingsOpen(false)}
         config={scoringConfig}
         onSave={(newConfig) => {
           setScoringConfig(newConfig);
           // Trigger re-analysis if we have results
           if (appState === AppState.RESULTS && manuscriptText && bibliographyText) {
              handleAnalysis(newConfig);
           }
         }}
       />

      {/* Main Content */}
      <main className="pt-28 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto min-h-screen flex flex-col">
        
        {appState === AppState.IDLE && (
          <div className="animate-fade-in-up w-full max-w-5xl mx-auto flex-1 flex flex-col justify-center">
            
            {/* Hero Section */}
            <div className="text-center mb-12 space-y-4">
              <h1 className="text-4xl md:text-6xl font-display font-bold text-white tracking-tight leading-tight">
                Academic Reference Analysis
              </h1>
              <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed font-normal">
                Evaluate relevance, recency, and authority of your manuscript's citation network.
              </p>
            </div>

            {/* Upload Section */}
            <div className="glass-card rounded-3xl p-2 md:p-3 shadow-2xl shadow-brand-900/5 ring-1 ring-white/10">
               <div className="bg-slate-900/50 rounded-2xl border border-slate-700/50 overflow-hidden">
                 <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-700/50">
                    <div className="p-8 md:p-12 hover:bg-slate-800/40 transition-colors duration-500">
                      <FileUpload 
                        label="Manuscript" 
                        subLabel="Paste or upload .tex content" 
                        icon={Icons.Manuscript}
                        value={manuscriptText}
                        onChange={setManuscriptText}
                        accept=".tex,.txt,.md"
                      />
                    </div>
                    <div className="p-8 md:p-12 hover:bg-slate-800/40 transition-colors duration-500">
                      <FileUpload 
                        label="Bibliography" 
                        subLabel="Paste or upload .bib content" 
                        icon={Icons.Bibliography}
                        value={bibliographyText}
                        onChange={setBibliographyText}
                        accept=".bib,.json,.csv,.txt"
                      />
                      <div className="mt-6 flex items-center justify-between p-3 rounded-xl border border-slate-700/60 bg-slate-900/40">
                        <div className="space-y-0.5">
                          <div className="text-sm font-bold text-slate-200">Generate References Automatically</div>
                          <div className="text-xs text-slate-400">No .bib file? Use DeepSearch to build citations and BibTeX</div>
                        </div>
                        <button
                          type="button"
                          onClick={() => setAutoGenerateBib(prev => !prev)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            autoGenerateBib ? 'bg-brand-500' : 'bg-slate-700'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              autoGenerateBib ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>
                    </div>
                 </div>
                 
                 <div className="bg-slate-900/40 backdrop-blur-sm p-8 flex flex-col items-center border-t border-slate-700/50">
                    {errorMsg && (
                      <div className="mb-6 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-center gap-3 text-sm font-medium animate-shake shadow-sm">
                        <Icons.Warning className="w-5 h-5 flex-shrink-0" />
                        {errorMsg}
                      </div>
                    )}

                    {autoGenerateBib && (
                      <div className="mb-4 px-3 py-2 text-xs text-slate-300 bg-slate-800/50 rounded-lg border border-slate-700/60">
                        DeepSearch will refine your manuscript and generate a bibliography automatically.
                      </div>
                    )}

                    <button
                      onClick={() => handleAnalysis()}
                      disabled={!manuscriptText || (!bibliographyText && !autoGenerateBib)}
                      className={`
                        group relative inline-flex items-center justify-center px-10 py-4 text-lg font-bold text-white transition-all duration-300 
                        rounded-2xl shadow-xl shadow-brand-500/30 focus:outline-none focus:ring-4 focus:ring-brand-500/20 active:scale-95
                        ${(!manuscriptText || (!bibliographyText && !autoGenerateBib)) 
                          ? 'bg-slate-800 cursor-not-allowed shadow-none opacity-50' 
                          : 'bg-brand-600 hover:bg-brand-500 hover:shadow-brand-600/40 hover:-translate-y-1'
                        }
                      `}
                    >
                      Analyze Documents
                      <Icons.ArrowRight className={`ml-2 w-5 h-5 transition-transform duration-300 ${(!manuscriptText || (!bibliographyText && !autoGenerateBib)) ? '' : 'group-hover:translate-x-1'}`} />
                    </button>
                 </div>
               </div>
            </div>

            {/* Features Footer */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-24 opacity-80">
               {[
                 { icon: Icons.Activity, color: 'text-brand-400', bg: 'bg-brand-500/10', title: 'Contextual Analysis', desc: 'Evaluates how you use citations, not just that they exist.' },
                 { icon: Icons.Chart, color: 'text-slate-200', bg: 'bg-slate-700/50', title: 'Visual Metrics', desc: 'Get a 5-dimensional score for every reference in your paper.' },
                 { icon: Icons.Warning, color: 'text-red-400', bg: 'bg-red-500/10', title: 'Gap Detection', desc: 'Identifies unsupported claims and outdated sources.' }
               ].map((feature, i) => (
                 <div key={i} className="flex flex-col items-center text-center gap-4 group cursor-default">
                   <div className={`p-4 ${feature.bg} ${feature.color} rounded-2xl shadow-sm group-hover:scale-110 transition-transform duration-300`}>
                     <feature.icon className="w-6 h-6" />
                   </div>
                   <h3 className="font-display font-bold text-lg text-slate-100 group-hover:text-brand-400 transition-colors">{feature.title}</h3>
                   <p className="text-sm text-slate-500 leading-relaxed max-w-xs">{feature.desc}</p>
                 </div>
               ))}
            </div>

          </div>
        )}

        {(appState === AppState.PARSING || appState === AppState.EMBEDDING || appState === AppState.SCORING) && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] animate-fade-in flex-1">
            <div className="w-full max-w-md space-y-4">
              <div className="flex justify-between items-center text-sm font-medium text-slate-400">
                <span>{deepSearchProgress > 0 ? "Refining Document" : "Processing Content"}</span>
                <span className="text-brand-400">
                    {deepSearchProgress > 0 
                        ? `${Math.round(deepSearchProgress)}%` 
                        : `${Math.round(appState === AppState.PARSING ? 33 : appState === AppState.EMBEDDING ? 66 : 90)}%`
                    }
                </span>
              </div>
              <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                 <div 
                    className="h-full bg-brand-600 transition-all duration-300 ease-out rounded-full"
                    style={{ 
                        width: deepSearchProgress > 0 
                            ? `${deepSearchProgress}%` 
                            : `${appState === AppState.PARSING ? 33 : appState === AppState.EMBEDDING ? 66 : 100}%` 
                    }}
                 ></div>
              </div>
              <p className="text-slate-500 text-sm text-center animate-pulse">{statusMessage}</p>
            </div>
          </div>
        )}

        {appState === AppState.RESULTS && result && (
           <div className="animate-fade-in w-full">
             {activeTab === 'dashboard' ? (
                <ResultsDashboard result={result} onReset={handleReset} scoringConfig={scoringConfig} />
             ) : activeTab === 'document' ? (
                <DocumentViewer 
                    result={result} 
                    onReset={handleReset} 
                    onUpdate={handleUpdate}
                    manuscriptText={manuscriptText}
                    bibliographyText={bibliographyText}
                    scoringConfig={scoringConfig}
                />
             ) : (
                <BibliographyDashboard
                  result={result}
                  manuscriptText={manuscriptText}
                  bibliographyText={bibliographyText}
                  onUpdate={handleUpdate}
                  scoringConfig={scoringConfig}
                />
             )}
           </div>
        )}
        
        {appState === AppState.ERROR && (
           <div className="flex flex-col items-center justify-center min-h-[60vh] text-center max-w-lg mx-auto flex-1 animate-scale-in">
             <div className="w-24 h-24 bg-red-50 rounded-3xl flex items-center justify-center mb-8 shadow-xl shadow-red-500/10 border border-red-100 rotate-3 hover:rotate-0 transition-transform duration-300">
               <Icons.Warning className="w-12 h-12 text-red-500" />
             </div>
             <h2 className="text-3xl font-display font-bold text-slate-900 mb-4">Analysis Failed</h2>
             <p className="text-slate-600 mb-10 leading-relaxed text-lg">{errorMsg || "An unexpected error occurred while processing your documents."}</p>
             <button 
               onClick={handleReset}
               className="px-8 py-3.5 bg-white text-slate-700 border border-slate-200 rounded-xl font-bold hover:bg-slate-50 hover:text-slate-900 hover:border-slate-300 transition-all shadow-sm hover:shadow-md"
             >
               Return Home
             </button>
           </div>
        )}

      </main>
    </div>
  );
}

export default App;
