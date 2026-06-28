import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, FlaskConical, Zap, BookOpen, SlidersHorizontal } from "lucide-react";

import ImagePicker from "./components/ImagePicker";
import ParamPanel from "./components/ParamPanel";
import StageStrip from "./components/StageStrip";
import FeatureInspector from "./components/FeatureInspector";
import ModelCompare from "./components/ModelCompare";

import {
  fetchDemos, runPipeline,
  DEFAULT_PARAMS,
  type DemoImage, type PipelineParams, type RunResult,
} from "./lib/api";

// ── debounce hook ──────────────────────────────────────────────────────────────
function useDebounce<T>(value: T, delay: number): T {
  const [dv, setDv] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDv(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return dv;
}

const SECTIONS = [
  { id: "pipeline", label: "Pipeline", icon: FlaskConical },
  { id: "features", label: "Features", icon: BookOpen    },
  { id: "models",   label: "Models",   icon: Zap         },
];

const SEVERITY: Record<string, { text: string; emoji: string; color: string }> = {
  acne1: { text: "Mild — Grade I",      emoji: "🟢", color: "text-emerald-700" },
  acne2: { text: "Moderate — Grade II", emoji: "🟡", color: "text-amber-700"   },
  acne3: { text: "Severe — Grade III",  emoji: "🔴", color: "text-rose-700"    },
};

export default function App() {
  const [demos,          setDemos]          = useState<DemoImage[]>([]);
  const [selectedDemoId, setSelectedDemoId] = useState<string | null>(null);
  const [uploadedFile,   setUploadedFile]   = useState<File | null>(null);
  const [params,         setParams]         = useState<PipelineParams>(DEFAULT_PARAMS);
  const [result,         setResult]         = useState<RunResult | null>(null);
  const [loading,        setLoading]        = useState(false);
  const [error,          setError]          = useState<string | null>(null);
  const [activeSection,  setActiveSection]  = useState("pipeline");
  const [paramsOpen,     setParamsOpen]     = useState(false);

  const debouncedParams = useDebounce(params, 280);

  // Load demo gallery
  useEffect(() => {
    fetchDemos()
      .then(setDemos)
      .catch(() => setError("Could not reach backend — is uvicorn running on :8000?"));
  }, []);

  // Re-run when image or params change
  const hasImage = selectedDemoId || uploadedFile;
  useEffect(() => {
    if (!hasImage) return;
    setLoading(true);
    setError(null);
    runPipeline({
      demoId: selectedDemoId ?? undefined,
      file:   uploadedFile   ?? undefined,
      params: debouncedParams,
    })
      .then(r  => { setResult(r); setLoading(false); })
      .catch(e => { if (e.name !== "AbortError") { setError(String(e.message ?? e)); setLoading(false); } });
  }, [selectedDemoId, uploadedFile, debouncedParams]);

  const trace    = result?.trace       ?? null;
  const preds    = result?.predictions ?? [];
  const champion = preds.find(p => p.is_champion);
  const champLabel = champion?.label;

  return (
    <div className="min-h-screen bg-[#fafaf8]">

      {/* ── Hero header ────────────────────────────────────────────────────── */}
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-6xl mx-auto px-5 py-10">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-2xl bg-teal-800 flex items-center justify-center text-white text-2xl shadow-lg shadow-teal-200 flex-none mt-1 animate-float">
              🔬
            </div>
            <div className="flex-1">
              <h1 className="font-display font-extrabold text-3xl md:text-4xl text-gray-950 tracking-tight leading-tight">
                Acne Detection
                <span className="text-orange-600"> CV Playground</span>
              </h1>
              <p className="mt-2 text-[15px] text-gray-500 max-w-xl leading-relaxed">
                Upload a face photo (or pick a demo) and watch the real ML pipeline run step-by-step —
                from raw pixels to a 42-dim feature vector to severity prediction. Tweak every parameter live.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {[
                  { icon: "🎛️", text: "14 live sliders"           },
                  { icon: "🧮", text: "42-dim feature inspector"   },
                  { icon: "🤖", text: `${preds.length || "18+"}  classifiers`},
                  { icon: "🔁", text: "Real OpenCV + sklearn"      },
                ].map(b => (
                  <span key={b.text} className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-gray-100 text-xs font-semibold text-gray-600">
                    {b.icon} {b.text}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* ── Error banner ───────────────────────────────────────────────────── */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            className="bg-rose-50 border-b border-rose-200 px-5 py-3 text-sm text-rose-700 font-medium text-center"
          >
            ⚠️ {error}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="max-w-6xl mx-auto px-5 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-8 items-start">

          {/* ── LEFT column ─────────────────────────────────────────────── */}
          <div className="space-y-8 min-w-0">

            {/* Step 1: Image picker */}
            <div className="rounded-2xl border border-gray-200 bg-white shadow-sm p-5">
              <div className="flex items-center gap-2 mb-5">
                <span className="w-6 h-6 rounded-md bg-violet-600 flex items-center justify-center text-white text-xs font-bold">1</span>
                <h2 className="font-display font-bold text-sm text-gray-900">Choose an image</h2>
              </div>
              <ImagePicker
                demos={demos}
                selectedDemoId={selectedDemoId}
                onSelectDemo={id => { setUploadedFile(null); setSelectedDemoId(id); }}
                onUpload={f  => { setSelectedDemoId(null); setUploadedFile(f); }}
                loading={loading}
              />
            </div>

            {/* Result / loading / empty state */}
            <AnimatePresence mode="wait">
              {loading && (
                <motion.div key="loading"
                  initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                  className="rounded-2xl border border-violet-200 bg-violet-50 px-6 py-8 flex flex-col items-center gap-3 text-center"
                >
                  <Loader2 className="w-8 h-8 text-violet-500 animate-spin" />
                  <p className="font-display font-bold text-violet-800">Running pipeline…</p>
                  <p className="text-xs text-violet-500">OpenCV preprocessing → 42-dim features → all classifiers</p>
                </motion.div>
              )}

              {!loading && champLabel && (
                <motion.div key="result"
                  initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="rounded-2xl border border-gray-200 bg-white shadow-sm px-6 py-5"
                >
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                      <p className="text-[11px] font-bold text-gray-400 uppercase tracking-widest mb-1">Champion (SGD) predicts</p>
                      <div className="flex items-baseline gap-3">
                        <span className="text-3xl">{SEVERITY[champLabel]?.emoji}</span>
                        <span className={`font-display font-extrabold text-2xl ${SEVERITY[champLabel]?.color}`}>
                          {SEVERITY[champLabel]?.text}
                        </span>
                      </div>
                      {champion?.confidence != null && (
                        <p className="text-xs text-gray-400 mt-1">
                          Confidence <span className="font-mono font-bold text-gray-600">{(champion.confidence * 100).toFixed(1)}%</span>
                        </p>
                      )}
                    </div>
                    <div className="text-right space-y-1">
                      <p className="text-[11px] text-gray-400">Skin pixels</p>
                      <p className="font-mono font-bold text-gray-700">{trace?.skin_pixel_count?.toLocaleString()}</p>
                      <p className="text-[11px] text-gray-400">Lesion blobs kept</p>
                      <p className="font-mono font-bold text-gray-700">{trace?.kept_blobs}</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {!loading && !hasImage && (
                <motion.div key="empty"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="rounded-2xl border-2 border-dashed border-gray-200 px-6 py-14 text-center"
                >
                  <p className="font-display text-4xl mb-3">👆</p>
                  <p className="font-display font-bold text-gray-400">Pick or upload an image above to start</p>
                  <p className="text-xs text-gray-300 mt-1">The full 12-stage pipeline runs automatically</p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Section nav (sticky) */}
            {hasImage && (
              <div className="flex gap-1 rounded-xl bg-gray-100 p-1 sticky top-4 z-10 shadow-sm">
                {SECTIONS.map(s => {
                  const Icon = s.icon;
                  return (
                    <button key={s.id}
                      onClick={() => {
                        setActiveSection(s.id);
                        document.getElementById(s.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
                      }}
                      className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-semibold transition-all
                        ${activeSection === s.id ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"}`}
                    >
                      <Icon className="w-3.5 h-3.5" /> {s.label}
                    </button>
                  );
                })}
              </div>
            )}

            {/* Pipeline stages */}
            <div id="pipeline"><StageStrip trace={trace} /></div>

            {/* Feature inspector */}
            <div id="features"><FeatureInspector trace={trace} /></div>

            {/* Model compare */}
            <div id="models"><ModelCompare predictions={preds} /></div>
          </div>

          {/* ── RIGHT: sticky param panel + stats ──────────────────────── */}
          <div className="lg:sticky lg:top-6 space-y-3">

            {/* Mobile toggle */}
            <button
              onClick={() => setParamsOpen(o => !o)}
              className="lg:hidden w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-gray-200 bg-white text-sm font-semibold text-gray-700 shadow-sm"
            >
              <SlidersHorizontal className="w-4 h-4 text-violet-500" />
              {paramsOpen ? "Hide" : "Show"} parameters
            </button>

            <div className={`lg:block ${paramsOpen ? "block" : "hidden"}`}>
              <ParamPanel params={params} onChange={setParams} />
            </div>

            {/* Pipeline stats card */}
            {trace && (
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="rounded-2xl border border-gray-200 bg-white shadow-sm p-4 space-y-3"
              >
                <h3 className="font-display font-bold text-[11px] text-gray-400 uppercase tracking-widest">Pipeline stats</h3>
                {([
                  ["Face detected",  trace.face_detected ? `Yes (attempt ${trace.cascade_attempt + 1})` : "No — full frame"],
                  ["Skin pixels",    trace.skin_pixel_count.toLocaleString()],
                  ["ROI exclusions", String(trace.roi_regions.length)],
                  ["Total blobs",    String(trace.blobs.length)],
                  ["Kept blobs",     String(trace.kept_blobs)],
                  ["thr Cr",         trace.thr_cr.toFixed(1)],
                  ["thr a*",         trace.thr_a.toFixed(1)],
                  ["μ Cr",           trace.mu_cr.toFixed(2)],
                  ["μ a*",           trace.mu_a.toFixed(2)],
                ] as [string, string][]).map(([k, v]) => (
                  <div key={k} className="flex justify-between items-center text-[12px]">
                    <span className="text-gray-500">{k}</span>
                    <span className="font-mono font-bold text-gray-800">{v}</span>
                  </div>
                ))}
              </motion.div>
            )}
          </div>
        </div>
      </div>

      <footer className="mt-16 border-t border-gray-200 py-6 px-5">
        <div className="max-w-6xl mx-auto flex items-center justify-between text-xs text-gray-400 flex-wrap gap-2">
          <span>Acne CV Playground · ACNE04 · 42-dim handcrafted features</span>
          <span className="font-mono">images processed in-memory · never stored</span>
        </div>
      </footer>
    </div>
  );
}
