import React, { useRef, useState } from 'react';
import { Icons } from './Icons';

interface FileUploadProps {
  label: string;
  subLabel: string;
  icon: React.ElementType;
  value: string;
  onChange: (text: string) => void;
  accept: string;
}

export const FileUpload: React.FC<FileUploadProps> = ({ 
  label, 
  subLabel, 
  icon: Icon, 
  value, 
  onChange,
  accept 
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFileName(file.name);
      const text = await file.text();
      onChange(text);
    }
  };

  const handlePaste = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
    setFileName(null);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files?.[0];
    if (file && accept.includes(file.name.split('.').pop() || '')) {
      setFileName(file.name);
      const text = await file.text();
      onChange(text);
    }
  };

  return (
    <div 
      className={`
        flex flex-col h-[320px] w-full rounded-2xl transition-all duration-300 relative overflow-hidden group
        ${isFocused || value ? 'glass-card shadow-xl shadow-black/20 ring-1 ring-brand-500/20' : 'bg-slate-900/30 border border-dashed border-slate-700 hover:border-brand-500/50 hover:bg-slate-900/50'}
        ${isDragging ? 'ring-2 ring-brand-500/50 scale-[1.02] border-brand-500 bg-brand-500/10' : ''}
      `}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Header */}
      <div className={`
        px-5 py-4 border-b flex items-center justify-between transition-colors duration-300
        ${isFocused || value ? 'bg-slate-900/50 border-slate-800' : 'bg-transparent border-slate-800'}
      `}>
        <div className="flex items-center gap-3">
          <div className={`
            p-2 rounded-xl transition-all duration-300
            ${isFocused || value ? 'bg-brand-500 text-slate-100 shadow-lg shadow-brand-500/30' : 'bg-slate-800 text-slate-400 group-hover:bg-brand-500/20 group-hover:text-brand-400'}
          `}>
            <Icon className="w-4 h-4" />
          </div>
          <div>
            <span className={`block text-sm font-bold transition-colors ${isFocused || value ? 'text-slate-100' : 'text-slate-400 group-hover:text-slate-100'}`}>
              {label}
            </span>
            <span className="block text-[10px] text-slate-500 font-medium uppercase tracking-wider">
              {subLabel}
            </span>
          </div>
        </div>
        <button 
          onClick={() => fileInputRef.current?.click()}
          className="text-xs font-bold text-brand-400 hover:text-slate-100 bg-brand-500/10 hover:bg-brand-500 border border-brand-500/20 hover:border-brand-500 px-3 py-1.5 rounded-lg transition-all duration-200 flex items-center gap-1.5"
        >
          <Icons.Upload className="w-3 h-3" />
          Import
        </button>
      </div>
      
      {/* Content Area */}
      <div className="flex-1 relative">
        {!value && !isFocused && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-500">
             <div className="text-center text-slate-500">
                <Icons.Upload className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p className="text-sm font-medium">Drag & Drop or Paste Content</p>
             </div>
          </div>
        )}
        
        {value ? (
           <div className="absolute inset-0 flex items-center justify-center p-6 animate-fade-in">
              <div className="text-center">
                  <div className="w-16 h-16 mx-auto bg-brand-500/10 rounded-full flex items-center justify-center mb-4 text-brand-400 border border-brand-500/20 shadow-lg shadow-brand-500/10">
                      <Icons.Success className="w-8 h-8" />
                  </div>
                  <h3 className="text-slate-100 font-display font-bold text-lg mb-2">
                      {fileName ? 'File Uploaded Successfully' : 'Content Added Successfully'}
                  </h3>
                  <p className="text-slate-400 text-sm font-medium max-w-[250px] mx-auto truncate mb-6">
                      {fileName || `${value.length.toLocaleString()} characters`}
                  </p>
                  <button 
                     onClick={(e) => { e.stopPropagation(); setFileName(null); onChange(''); }}
                     className="text-xs font-bold text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 hover:border-slate-600 px-4 py-2 rounded-lg transition-all duration-200"
                  >
                    Remove & Replace
                  </button>
              </div>
           </div>
        ) : (
            <textarea
              className="w-full h-full p-5 resize-none text-sm text-slate-300 leading-relaxed focus:outline-none font-mono bg-transparent relative z-10 placeholder:text-slate-600"
              placeholder={isFocused ? "Paste content here..." : ""}
              value={value}
              onChange={handlePaste}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
            />
        )}
        
        <input 
          type="file" 
          ref={fileInputRef}
          className="hidden"
          accept={accept}
          onChange={handleFileChange}
        />
      </div>
      
      {/* Visual Corner Accent */}
      {(isFocused || value) && (
        <div className="absolute bottom-0 right-0 w-16 h-16 bg-gradient-to-tl from-brand-500/10 to-transparent opacity-50 pointer-events-none rounded-br-2xl"></div>
      )}
    </div>
  );
};