import React from 'react';
import { ArrowRight, CheckCircle, Zap, Search, BookOpen, BarChart3, ShieldCheck } from 'lucide-react';

interface LandingPageProps {
  onStart: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onStart }) => {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-brand-500/30 selection:text-brand-100 overflow-x-hidden">
      {/* Navigation */}
      <nav className="fixed w-full z-50 glass border-b border-slate-800 bg-slate-950/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center gap-3">
              <div className="bg-brand-600 p-2 rounded-lg shadow-lg shadow-brand-500/20">
                <BookOpen className="w-6 h-6 text-slate-950" />
              </div>
              <span className="font-display font-bold text-xl tracking-tight text-slate-100">
                Ref<span className="text-brand-500">Score</span>
              </span>
            </div>
            <div className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-sm font-medium text-slate-400 hover:text-brand-400 transition-colors">Features</a>
              <a href="#benefits" className="text-sm font-medium text-slate-400 hover:text-brand-400 transition-colors">Benefits</a>
              <a href="#how-it-works" className="text-sm font-medium text-slate-400 hover:text-brand-400 transition-colors">How it Works</a>
              <button 
                onClick={onStart}
                className="px-5 py-2.5 bg-brand-600 hover:bg-brand-500 text-slate-950 font-bold rounded-lg transition-all duration-300 shadow-lg shadow-brand-500/20 hover:shadow-brand-500/40 hover:-translate-y-0.5"
              >
                Launch App
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-700 text-brand-400 text-xs font-semibold uppercase tracking-wider mb-8 animate-fade-in">
            <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse" />
            New Generation Reference Analysis
          </div>
          
          <h1 className="text-5xl md:text-7xl font-display font-bold text-slate-100 mb-8 leading-tight tracking-tight animate-fade-in-up">
            Master Your <span className="text-brand-500">Academic References</span>
          </h1>
          
          <p className="max-w-2xl mx-auto text-lg md:text-xl text-slate-400 mb-10 leading-relaxed animate-fade-in-up delay-100">
            Ensure citation accuracy, relevance, and authority with our advanced multi-dimensional scoring engine. 
            Stop guessing and start validating your research sources.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in-up delay-200">
            <button 
              onClick={onStart}
              className="w-full sm:w-auto px-8 py-4 bg-brand-600 hover:bg-brand-500 text-slate-950 font-bold text-lg rounded-xl transition-all duration-300 shadow-xl shadow-brand-500/20 hover:shadow-brand-500/40 hover:-translate-y-1 flex items-center justify-center gap-2"
            >
              Start Analyzing Now <ArrowRight className="w-5 h-5" />
            </button>
            <a 
              href="#features"
              className="w-full sm:w-auto px-8 py-4 bg-slate-900 hover:bg-slate-800 text-slate-200 font-semibold text-lg rounded-xl border border-slate-700 transition-all duration-300 hover:border-brand-500/50"
            >
              Explore Features
            </a>
          </div>
        </div>
      </section>

      {/* Feature Highlights */}
      <section id="features" className="py-24 bg-slate-900/50 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl md:text-4xl font-display font-bold text-slate-100 mb-4">
              Powerful Tools for Researchers
            </h2>
            <p className="text-slate-400 text-lg">
              Everything you need to validate, organize, and improve your bibliography in one cohesive platform.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard 
              icon={<Search className="w-8 h-8 text-brand-500" />}
              title="Semantic Analysis"
              description="Our engine understands the context of your citations, ensuring they align perfectly with your arguments."
            />
            <FeatureCard 
              icon={<BarChart3 className="w-8 h-8 text-red-500" />}
              title="Impact Scoring"
              description="Automatically evaluate the authority and recency of your sources with our proprietary scoring algorithm."
            />
            <FeatureCard 
              icon={<Zap className="w-8 h-8 text-brand-400" />}
              title="Instant Validation"
              description="Detect missing metadata, formatting errors, and potential hallucinations in real-time."
            />
          </div>
        </div>
      </section>

      {/* Benefits Overview */}
      <section id="benefits" className="py-24 relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="relative">
              <div className="relative bg-slate-900 border border-slate-700 rounded-2xl p-8 shadow-2xl">
                <div className="space-y-6">
                    <div className="flex items-center gap-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                        <div className="w-12 h-12 bg-slate-700/50 rounded-full flex items-center justify-center border border-slate-600/30">
                            <span className="text-brand-400 font-bold">98</span>
                        </div>
                        <div>
                            <h4 className="font-semibold text-slate-200">Alignment Score</h4>
                            <p className="text-sm text-slate-400">Perfect contextual match</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                        <div className="w-12 h-12 bg-brand-900/30 rounded-full flex items-center justify-center border border-brand-500/20">
                            <span className="text-brand-500 font-bold">A+</span>
                        </div>
                        <div>
                            <h4 className="font-semibold text-slate-200">Authority Grade</h4>
                            <p className="text-sm text-slate-400">High impact sources detected</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                        <div className="w-12 h-12 bg-red-900/30 rounded-full flex items-center justify-center border border-red-500/20">
                            <ShieldCheck className="w-6 h-6 text-red-500" />
                        </div>
                        <div>
                            <h4 className="font-semibold text-slate-200">Compliance Check</h4>
                            <p className="text-sm text-slate-400">0 Formatting errors found</p>
                        </div>
                    </div>
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-3xl md:text-4xl font-display font-bold text-slate-100 mb-6">
                Why Choose RefScore?
              </h2>
              <p className="text-slate-400 text-lg mb-8">
                Academic writing is rigorous. Your tools should be too. RefScore bridges the gap between manual checking and automated perfection.
              </p>
              
              <div className="space-y-4">
                <BenefitItem text="Save hours of manual verification time" />
                <BenefitItem text="Improve paper acceptance rates with better sources" />
                <BenefitItem text="Ensure consistent formatting across all references" />
                <BenefitItem text="Discover more relevant papers automatically" />
              </div>

              <div className="mt-10">
                <button 
                  onClick={onStart}
                  className="px-6 py-3 bg-slate-100 text-slate-950 font-bold rounded-lg hover:bg-white transition-colors shadow-lg shadow-white/10"
                >
                  Get Started for Free
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 relative bg-slate-900">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <h2 className="text-4xl md:text-5xl font-display font-bold text-slate-100 mb-8">
            Ready to elevate your research?
          </h2>
          <p className="text-xl text-slate-400 mb-12 max-w-2xl mx-auto">
            Join the new standard of academic rigor. No sign-up required for the demo version.
          </p>
          <button 
            onClick={onStart}
            className="px-10 py-5 bg-brand-600 hover:bg-brand-500 text-slate-950 font-bold text-xl rounded-xl transition-all duration-300 shadow-2xl shadow-brand-500/30 hover:shadow-brand-500/50 hover:scale-105"
          >
            Launch RefScore Now
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-slate-800 bg-slate-950 text-center">
        <div className="max-w-7xl mx-auto px-4 text-slate-500">
          <p>&copy; {new Date().getFullYear()} RefScore. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

const FeatureCard = ({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) => (
  <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800 hover:border-brand-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-brand-500/5 group">
    <div className="mb-6 p-3 bg-slate-950 rounded-xl inline-block border border-slate-800 group-hover:border-brand-500/20 transition-colors">
      {icon}
    </div>
    <h3 className="text-xl font-bold text-slate-100 mb-3">{title}</h3>
    <p className="text-slate-400 leading-relaxed">
      {description}
    </p>
  </div>
);

const BenefitItem = ({ text }: { text: string }) => (
  <div className="flex items-center gap-3">
    <CheckCircle className="w-5 h-5 text-brand-500 shrink-0" />
    <span className="text-slate-300">{text}</span>
  </div>
);
