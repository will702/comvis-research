import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { Feature, Trace } from "../lib/api";

const GROUP_META: Record<string, { color: string; bg: string; border: string; desc: string }> = {
  "Structural (8)":     { color: "text-blue-700",   bg: "bg-blue-50",   border: "border-blue-200",   desc: "Lesion count, area, intensity stats — the 'how many / how big / how red' summary" },
  "LBP R=1 (9)":        { color: "text-indigo-700", bg: "bg-indigo-50", border: "border-indigo-200", desc: "Fine-scale texture: LBP uniform histogram at radius 1 (single-pixel neighbourhoods)" },
  "LBP R=2 (9)":        { color: "text-violet-700", bg: "bg-violet-50", border: "border-violet-200", desc: "Medium-scale texture: LBP at radius 2 — captures pore / small-lesion patterns" },
  "LBP R=3 (9)":        { color: "text-purple-700", bg: "bg-purple-50", border: "border-purple-200", desc: "Coarse texture: LBP at radius 3 — broader surface roughness across skin" },
  "GLCM (3)":           { color: "text-cyan-700",   bg: "bg-cyan-50",   border: "border-cyan-200",   desc: "Grey co-occurrence: contrast, homogeneity, energy — ordered surface vs. rough inflamed skin" },
  "Global Redness (4)": { color: "text-rose-700",   bg: "bg-rose-50",   border: "border-rose-200",   desc: "Mean + std of LAB a* and YCrCb Cr over all skin — whole-face inflammatory level" },
};

interface Props { trace: Trace | null }

export default function FeatureInspector({ trace }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);
  const [pinnedGroup, setPinnedGroup] = useState<string | null>(null);

  const features: Feature[] = trace?.features ?? [];
  const groups: Record<string, number[]> = trace?.feature_groups ?? {};

  // Build max-value per group for bar scaling
  const groupMaxAbs: Record<string, number> = {};
  for (const [gName, idxs] of Object.entries(groups)) {
    const vals = idxs.map(i => Math.abs(features[i]?.value ?? 0));
    groupMaxAbs[gName] = Math.max(...vals, 1e-9);
  }

  const hoveredFeature = hovered !== null ? features[hovered] : null;

  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-100 flex items-start justify-between gap-4">
        <div>
          <h2 className="font-display font-bold text-base text-gray-900">42-Dim Feature Vector</h2>
          <p className="text-[12px] text-gray-400 mt-0.5">Hover any cell to see what it means and where it came from</p>
        </div>
        <div className="flex-none font-mono text-xs text-gray-300 bg-gray-50 border border-gray-100 rounded-lg px-2 py-1.5 mt-0.5">
          {features.length}/42 dims
        </div>
      </div>

      {/* Tooltip / detail pane */}
      <AnimatePresence mode="wait">
        {hoveredFeature ? (
          <motion.div
            key={hoveredFeature.idx}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="px-5 py-3 bg-gray-900 text-white"
          >
            <div className="flex items-baseline gap-3">
              <span className="font-mono text-xs text-gray-400">dim {hoveredFeature.idx}</span>
              <span className="font-mono text-sm font-bold text-white">{hoveredFeature.name}</span>
              <span className="ml-auto font-mono text-sm font-bold text-emerald-400">
                {hoveredFeature.value.toExponential(3)}
              </span>
            </div>
            <p className="text-[12px] text-gray-300 mt-1 leading-relaxed">{hoveredFeature.description}</p>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="px-5 py-3 bg-gray-50 border-b border-gray-100 min-h-[64px] flex items-center"
          >
            <p className="text-xs text-gray-300 italic">
              {features.length === 0 ? "Run the pipeline to populate the feature vector" : "Hover a cell ↓"}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Groups */}
      <div className="p-4 space-y-4">
        {Object.entries(groups).map(([gName, idxs]) => {
          const meta = GROUP_META[gName] ?? { color: "text-gray-700", bg: "bg-gray-50", border: "border-gray-200", desc: "" };
          const isHighlighted = pinnedGroup === null || pinnedGroup === gName;
          const maxAbs = groupMaxAbs[gName] ?? 1;

          return (
            <motion.div
              key={gName}
              className={`rounded-xl border p-3 transition-all cursor-pointer ${meta.border} ${isHighlighted ? meta.bg : "bg-gray-50 opacity-40"}`}
              onClick={() => setPinnedGroup(pinnedGroup === gName ? null : gName)}
              whileHover={{ scale: 1.005 }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className={`text-[11px] font-bold uppercase tracking-wider ${meta.color}`}>{gName}</span>
                <span className="text-[10px] text-gray-400">{idxs.length} dims</span>
              </div>
              <p className="text-[11px] text-gray-500 mb-3 leading-relaxed">{meta.desc}</p>

              {/* Cell grid */}
              <div className="flex flex-wrap gap-1.5">
                {idxs.map(i => {
                  const f = features[i];
                  const val = f?.value ?? 0;
                  const barH = Math.abs(val) / maxAbs;
                  const isHov = hovered === i;

                  return (
                    <motion.div
                      key={i}
                      className={`relative flex flex-col items-center rounded-lg overflow-hidden cursor-default border transition-all
                        ${isHov ? `border-current ${meta.border} shadow-md` : "border-transparent"}`}
                      style={{ width: 44, height: 52 }}
                      onMouseEnter={() => setHovered(i)}
                      onMouseLeave={() => setHovered(null)}
                      whileHover={{ scale: 1.12, zIndex: 10 }}
                    >
                      {/* Mini bar */}
                      <div className="w-full flex-1 relative bg-white rounded-t-lg overflow-hidden">
                        <motion.div
                          className={`absolute bottom-0 left-0 right-0 ${meta.bg} border-t ${meta.border}`}
                          initial={{ height: 0 }}
                          animate={{ height: `${Math.max(barH * 100, 2)}%` }}
                          transition={{ duration: 0.6, delay: i * 0.005, ease: "easeOut" }}
                        />
                      </div>
                      {/* Value label */}
                      <div className={`w-full py-0.5 text-center rounded-b-lg ${isHov ? meta.bg : "bg-white"}`}>
                        <span className={`font-mono text-[9px] font-bold ${meta.color} leading-none`}>
                          {Math.abs(val) >= 0.001 ? val.toFixed(2) : val.toExponential(0)}
                        </span>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          );
        })}

        {features.length === 0 && (
          <div className="py-12 text-center">
            <div className="font-display text-3xl mb-2">🧮</div>
            <p className="text-sm text-gray-400">Select an image and run the pipeline<br />to see the feature vector populate</p>
          </div>
        )}
      </div>
    </div>
  );
}
