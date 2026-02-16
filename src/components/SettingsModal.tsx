import React, { useState, useEffect } from 'react';
import { Icons } from './Icons';
import { ScoringConfig, ScoringEngine } from '../services/scoring/ScoringEngine';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: ScoringConfig;
  onSave: (newConfig: ScoringConfig) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, config, onSave }) => {
  const [weights, setWeights] = useState(config.weights);
  const [deepSearchEnabled, setDeepSearchEnabled] = useState(!!config.deepSearchEnabled);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    setWeights(config.weights);
    setDeepSearchEnabled(!!config.deepSearchEnabled);
  }, [config, isOpen]);

  useEffect(() => {
    const sum = Object.values(weights).reduce((acc, val) => acc + val, 0);
    setTotal(sum);
  }, [weights]);

  const handleChange = (key: keyof ScoringConfig['weights'], value: number) => {
    setWeights(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSave = () => {
    onSave({ weights, deepSearchEnabled });
    onClose();
  };

  const handleReset = () => {
    setWeights(ScoringEngine.DEFAULT_CONFIG.weights);
    setDeepSearchEnabled(!!ScoringEngine.DEFAULT_CONFIG.deepSearchEnabled);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-700 rounded-3xl w-full max-w-lg shadow-2xl animate-scale-in overflow-hidden">
        
        {/* Header */}
        <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
          <h2 className="text-xl font-display font-bold text-white flex items-center gap-2">
            <Icons.Settings className="w-5 h-5 text-brand-400" />
            Scoring Configuration
          </h2>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white transition-colors"
          >
            <Icons.X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 max-h-[70vh] overflow-y-auto custom-scrollbar">
          
          <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-xl">
             <div className="flex items-start gap-3">
               <Icons.Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
               <p className="text-sm text-blue-100/80 leading-relaxed">
                 Adjust the importance of each analysis dimension. The weights should ideally sum to 1.0 (100%) for accurate scoring.
               </p>
             </div>
          </div>

          <div className="space-y-5">
            {Object.entries(weights).map(([key, val]) => (
              <div key={key} className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-bold text-slate-300 capitalize">{key}</label>
                  <span className="text-xs font-mono font-bold text-brand-400 bg-brand-500/10 px-2 py-0.5 rounded border border-brand-500/20">
                    {(val * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={val}
                  onChange={(e) => handleChange(key as keyof ScoringConfig['weights'], parseFloat(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-brand-500 hover:accent-brand-400"
                />
              </div>
            ))}
          </div>

          

          {/* Total Indicator */}
          <div className={`flex justify-between items-center p-4 rounded-xl border ${
            Math.abs(total - 1.0) < 0.01 
              ? 'bg-green-500/10 border-green-500/20 text-green-400' 
              : 'bg-yellow-500/10 border-yellow-500/20 text-yellow-400'
          }`}>
             <span className="font-bold text-sm">Total Weight</span>
             <span className="font-mono font-bold">{(total * 100).toFixed(0)}%</span>
          </div>

        </div>

        {/* Footer */}
        <div className="p-6 border-t border-slate-800 bg-slate-900/50 flex justify-between items-center">
          <button 
            onClick={handleReset}
            className="text-sm font-bold text-slate-500 hover:text-slate-300 transition-colors"
          >
            Reset to Default
          </button>
          <div className="flex gap-3">
             <button 
                onClick={onClose}
                className="px-4 py-2 text-sm font-bold text-slate-300 hover:bg-slate-800 rounded-xl transition-colors"
             >
                Cancel
             </button>
             <button 
                onClick={handleSave}
                className="px-6 py-2 text-sm font-bold text-slate-900 bg-brand-400 hover:bg-brand-300 rounded-xl shadow-lg shadow-brand-500/20 transition-all active:scale-95"
             >
                Save Changes
             </button>
          </div>
        </div>

      </div>
    </div>
  );
};
