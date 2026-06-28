import { useState } from "react";
import { motion } from "framer-motion";
import { RotateCcw, ChevronDown, ChevronUp } from "lucide-react";
import type { PipelineParams } from "../lib/api";
import { DEFAULT_PARAMS } from "../lib/api";

interface SliderDef {
  key: keyof PipelineParams;
  label: string;
  min: number; max: number; step: number;
  tip: string;
  stage: string;
  color: string;
}

const SLIDERS: SliderDef[] = [
  // CLAHE
  { key: "clahe_clip",  label: "CLAHE Clip Limit", min: 1, max: 8, step: 0.1,
    tip: "Higher = more contrast boost. Too high creates noise halos.", stage: "Stage 3", color: "amber" },
  { key: "clahe_tile",  label: "CLAHE Tile Size",  min: 4, max: 16, step: 1,
    tip: "Grid tile size for local histogram equalization.", stage: "Stage 3", color: "amber" },
  // Skin seg
  { key: "skin_close", label: "Skin MORPH_CLOSE", min: 3, max: 19, step: 2,
    tip: "Closes holes in skin mask. Larger = more filled skin regions.", stage: "Stage 6", color: "emerald" },
  { key: "skin_open",  label: "Skin MORPH_OPEN",  min: 3, max: 11, step: 2,
    tip: "Removes noise in skin mask. Larger = cleaner but less skin.", stage: "Stage 6", color: "emerald" },
  // Lesion thresholds
  { key: "cr_k",    label: "Cr threshold × (σ)", min: 0.5, max: 3.0, step: 0.05,
    tip: "thr_cr = max(cr_floor, μCr + cr_k·σCr). Higher → stricter redness filter.", stage: "Stage 8", color: "rose" },
  { key: "a_k",     label: "a* threshold × (σ)", min: 0.5, max: 3.0, step: 0.05,
    tip: "thr_a = max(a_floor, μa* + a_k·σa*). Higher → stricter redness filter.", stage: "Stage 8", color: "rose" },
  { key: "cr_floor", label: "Cr floor value",     min: 130, max: 165, step: 1,
    tip: "Minimum Cr channel threshold regardless of skin stats.", stage: "Stage 8", color: "rose" },
  { key: "a_floor",  label: "a* floor value",     min: 120, max: 150, step: 1,
    tip: "Minimum LAB a* threshold regardless of skin stats.", stage: "Stage 8", color: "rose" },
  // Red mask morph
  { key: "red_open",  label: "Red mask OPEN",  min: 1, max: 9, step: 2,
    tip: "Removes small isolated red dots from red mask.", stage: "Stage 9", color: "pink" },
  { key: "red_close", label: "Red mask CLOSE", min: 3, max: 15, step: 2,
    tip: "Merges nearby red blobs into single lesions.", stage: "Stage 9", color: "pink" },
  // Shape filters
  { key: "max_aspect",  label: "Max aspect ratio",  min: 1.5, max: 6.0, step: 0.1,
    tip: "Reject blobs wider/taller than this ratio (e.g. wrinkles).", stage: "Stage 9", color: "violet" },
  { key: "min_fill",    label: "Min fill ratio",     min: 0.1, max: 0.6, step: 0.01,
    tip: "Reject hollow blobs: area / bbox must exceed this.", stage: "Stage 9", color: "violet" },
  { key: "max_area_pct", label: "Max blob size (%skin)", min: 0.01, max: 0.10, step: 0.005,
    tip: "Reject blobs > X% of skin area (diffuse redness zones).", stage: "Stage 9", color: "violet" },
  { key: "min_local_contrast", label: "Min local a* contrast", min: 0.5, max: 4.0, step: 0.1,
    tip: "Blob must be this much redder than its surrounding ring.", stage: "Stage 9", color: "violet" },
];

const COLOR_CLASSES: Record<string, { track: string; thumb: string; label: string; bg: string }> = {
  amber:  { track: "accent-amber-500",  thumb: "bg-amber-500",  label: "text-amber-700",  bg: "bg-amber-50"  },
  emerald:{ track: "accent-emerald-500",thumb: "bg-emerald-500",label: "text-emerald-700",bg: "bg-emerald-50"},
  rose:   { track: "accent-rose-500",   thumb: "bg-rose-500",   label: "text-rose-700",   bg: "bg-rose-50"   },
  pink:   { track: "accent-pink-500",   thumb: "bg-pink-500",   label: "text-pink-700",   bg: "bg-pink-50"   },
  violet: { track: "accent-violet-500", thumb: "bg-violet-500", label: "text-violet-700", bg: "bg-violet-50" },
};

interface Group { name: string; keys: (keyof PipelineParams)[]; color: string }
const GROUPS: Group[] = [
  { name: "CLAHE Enhancement",   keys: ["clahe_clip","clahe_tile"],           color: "amber"  },
  { name: "Skin Segmentation",   keys: ["skin_close","skin_open"],            color: "emerald"},
  { name: "Lesion Thresholds",   keys: ["cr_k","a_k","cr_floor","a_floor"],   color: "rose"   },
  { name: "Red Mask Morphology", keys: ["red_open","red_close"],              color: "pink"   },
  { name: "Shape Filters",       keys: ["max_aspect","min_fill","max_area_pct","min_local_contrast"], color: "violet"},
];

interface Props {
  params: PipelineParams;
  onChange: (p: PipelineParams) => void;
}

export default function ParamPanel({ params, onChange }: Props) {
  const [open, setOpen] = useState<Record<string, boolean>>({
    "CLAHE Enhancement": true,
    "Lesion Thresholds": true,
  });
  const [tip, setTip] = useState<string | null>(null);

  const sliderMap = Object.fromEntries(SLIDERS.map(s => [s.key, s]));

  function toggle(name: string) {
    setOpen(o => ({ ...o, [name]: !o[name] }));
  }

  function reset() {
    onChange({ ...DEFAULT_PARAMS });
  }

  function isDefault(key: keyof PipelineParams) {
    return params[key] === DEFAULT_PARAMS[key];
  }

  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
        <div>
          <h3 className="font-display font-bold text-sm text-gray-900">Tweak Parameters</h3>
          <p className="text-[11px] text-gray-400 mt-0.5">Sliders re-run the real pipeline live</p>
        </div>
        <motion.button
          onClick={reset}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-violet-600 hover:text-violet-800 hover:bg-violet-50 border border-violet-200 transition-colors"
          whileTap={{ scale: 0.95 }}
        >
          <RotateCcw className="w-3 h-3" /> Reset
        </motion.button>
      </div>

      {/* Tooltip bar */}
      <div className={`px-4 py-2 text-[11px] text-gray-500 bg-gray-50 border-b border-gray-100 min-h-[36px] transition-all ${tip ? "opacity-100" : "opacity-40"}`}>
        {tip ?? "Hover a slider for what it controls"}
      </div>

      <div className="divide-y divide-gray-100">
        {GROUPS.map(g => {
          const cc = COLOR_CLASSES[g.color];
          const isOpen = open[g.name] !== false;
          const modified = g.keys.some(k => !isDefault(k));
          return (
            <div key={g.name}>
              <button
                onClick={() => toggle(g.name)}
                className={`w-full px-4 py-2.5 flex items-center justify-between text-left transition-colors hover:${cc.bg}`}
              >
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-bold ${cc.label}`}>{g.name}</span>
                  {modified && (
                    <span className={`w-1.5 h-1.5 rounded-full ${cc.thumb}`} />
                  )}
                </div>
                {isOpen ? <ChevronUp className={`w-3.5 h-3.5 ${cc.label} opacity-60`} /> : <ChevronDown className={`w-3.5 h-3.5 ${cc.label} opacity-60`} />}
              </button>

              {isOpen && (
                <div className="px-4 pb-4 space-y-4">
                  {g.keys.map(key => {
                    const s = sliderMap[key as string];
                    if (!s) return null;
                    const val = params[key] as number;
                    const def = DEFAULT_PARAMS[key] as number;
                    const pct = ((val - s.min) / (s.max - s.min)) * 100;
                    return (
                      <div key={key as string} onMouseEnter={() => setTip(s.tip)} onMouseLeave={() => setTip(null)}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-[11px] font-semibold text-gray-700">{s.label}</span>
                          <div className="flex items-center gap-1.5">
                            {val !== def && (
                              <button
                                onClick={() => onChange({ ...params, [key]: def })}
                                className="text-[10px] text-gray-400 hover:text-violet-500 transition-colors"
                                title="Reset to default"
                              >↩</button>
                            )}
                            <span className={`font-mono text-xs font-bold ${cc.label}`}>
                              {Number.isInteger(s.step) ? val : val.toFixed(s.step < 0.1 ? 3 : 2)}
                            </span>
                          </div>
                        </div>
                        <div className="relative">
                          <input
                            type="range"
                            min={s.min} max={s.max} step={s.step}
                            value={val}
                            onChange={e => onChange({ ...params, [key]: parseFloat(e.target.value) })}
                            className={`w-full h-1.5 rounded-full appearance-none cursor-pointer ${cc.track}
                              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5
                              [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:${cc.thumb}
                              [&::-webkit-slider-thumb]:shadow-sm [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-white`}
                            style={{
                              background: `linear-gradient(to right, currentColor 0%, currentColor ${pct}%, #e5e7eb ${pct}%, #e5e7eb 100%)`,
                            }}
                          />
                          {/* Default marker */}
                          <div
                            className="absolute top-1/2 -translate-y-1/2 w-0.5 h-2.5 bg-gray-300 rounded-full pointer-events-none"
                            style={{ left: `${((def - s.min) / (s.max - s.min)) * 100}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-[10px] text-gray-300 mt-0.5">
                          <span>{s.min}</span><span>default: {def}</span><span>{s.max}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
