import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Trophy, ChevronDown, ChevronUp, AlertCircle } from "lucide-react";
import type { Prediction } from "../lib/api";

const LABEL_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  acne1: { bg: "bg-emerald-100", text: "text-emerald-800", dot: "bg-emerald-500" },
  acne2: { bg: "bg-amber-100",   text: "text-amber-800",   dot: "bg-amber-500"   },
  acne3: { bg: "bg-rose-100",    text: "text-rose-800",    dot: "bg-rose-500"    },
};

const SEVERITY: Record<string, string> = {
  acne1: "Mild",
  acne2: "Moderate",
  acne3: "Severe",
};

// Pretty display names
function prettyName(raw: string): string {
  return raw
    .replace(/_/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase())
    .replace("Svm", "SVM")
    .replace("Mlp", "MLP")
    .replace("Lda", "LDA")
    .replace("Qda", "QDA")
    .replace("Sgd", "SGD")
    .replace("Knn", "KNN")
    .replace("Nb", "NB");
}

interface Props {
  predictions: Prediction[];
}

export default function ModelCompare({ predictions }: Props) {
  const [showAll, setShowAll] = useState(false);
  const [sortBy, setSortBy] = useState<"f1" | "confidence" | "name">("f1");

  if (predictions.length === 0) {
    return (
      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm p-8 text-center">
        <div className="font-display text-3xl mb-2">🤖</div>
        <p className="text-sm text-gray-400">Run the pipeline to compare all {18}+ classifiers</p>
      </div>
    );
  }

  const champion = predictions.find(p => p.is_champion);
  const championLabel = champion?.label ?? "—";
  const lc = LABEL_COLORS[championLabel] ?? LABEL_COLORS.acne1;

  // Consensus: most common label
  const labelCounts: Record<string, number> = {};
  for (const p of predictions) if (p.label && p.label !== "error") labelCounts[p.label] = (labelCounts[p.label] ?? 0) + 1;
  const consensusLabel = Object.entries(labelCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—";
  const consensusPct   = Math.round((labelCounts[consensusLabel] ?? 0) / predictions.filter(p => p.label !== "error").length * 100);

  const sorted = [...predictions].sort((a, b) => {
    if (sortBy === "f1")         return (b.f1 ?? 0) - (a.f1 ?? 0);
    if (sortBy === "confidence") return (b.confidence ?? 0) - (a.confidence ?? 0);
    return a.name.localeCompare(b.name);
  });

  const displayed = showAll ? sorted : sorted.slice(0, 8);

  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-100">
        <h2 className="font-display font-bold text-base text-gray-900">Model Comparison</h2>
        <p className="text-[12px] text-gray-400 mt-0.5">All classifiers run on the same 42-dim vector</p>
      </div>

      {/* Summary banner */}
      <div className="px-5 py-4 bg-gray-50 border-b border-gray-100 grid grid-cols-2 gap-4">
        {/* Champion */}
        <div className="space-y-1">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold text-gray-500 uppercase tracking-wider">
            <Trophy className="w-3 h-3 text-amber-500" /> Champion (SGD)
          </div>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${lc.dot}`} />
            <span className="font-display font-bold text-xl text-gray-900">{SEVERITY[championLabel] ?? championLabel}</span>
          </div>
          {champion?.confidence != null && (
            <div className="text-[11px] text-gray-500">
              confidence <span className="font-mono font-bold text-gray-700">{(champion.confidence * 100).toFixed(1)}%</span>
            </div>
          )}
          {champion?.accuracy != null && (
            <div className="text-[11px] text-gray-400">
              test acc <span className="font-mono">{(champion.accuracy * 100).toFixed(1)}%</span>
              {champion.f1 != null && <> · F1 <span className="font-mono">{(champion.f1 * 100).toFixed(1)}%</span></>}
            </div>
          )}
        </div>

        {/* Consensus */}
        <div className="space-y-1 border-l border-gray-200 pl-4">
          <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Model consensus</div>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${LABEL_COLORS[consensusLabel]?.dot ?? "bg-gray-400"}`} />
            <span className="font-display font-bold text-xl text-gray-900">{SEVERITY[consensusLabel] ?? consensusLabel}</span>
          </div>
          <div className="text-[11px] text-gray-500">
            <span className="font-mono font-bold text-gray-700">{consensusPct}%</span> of models agree
          </div>
          <div className="flex gap-1.5 mt-1">
            {Object.entries(labelCounts).sort().map(([label, cnt]) => (
              <span key={label} className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${LABEL_COLORS[label]?.bg} ${LABEL_COLORS[label]?.text}`}>
                {label}: {cnt}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Sort controls */}
      <div className="px-5 pt-3 pb-1 flex items-center gap-3">
        <span className="text-[11px] text-gray-400 font-medium">Sort by</span>
        {(["f1", "confidence", "name"] as const).map(s => (
          <button
            key={s}
            onClick={() => setSortBy(s)}
            className={`text-[11px] font-semibold px-2 py-0.5 rounded-md transition-colors
              ${sortBy === s ? "bg-gray-900 text-white" : "text-gray-400 hover:text-gray-700"}`}
          >
            {s === "f1" ? "F1 score" : s === "confidence" ? "Confidence" : "Name"}
          </button>
        ))}
        <span className="ml-auto text-[11px] text-gray-300">{predictions.length} models</span>
      </div>

      {/* Model rows */}
      <div className="px-4 pb-2 space-y-1">
        <AnimatePresence initial={false}>
          {displayed.map((p, i) => {
            const lc2 = LABEL_COLORS[p.label] ?? { bg: "bg-gray-100", text: "text-gray-600", dot: "bg-gray-400" };
            const confPct = (p.confidence ?? 0) * 100;
            const f1Pct   = (p.f1 ?? 0) * 100;

            return (
              <motion.div
                key={p.name}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2, delay: i * 0.02 }}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl border transition-colors
                  ${p.is_champion ? "border-amber-200 bg-amber-50" : "border-transparent hover:bg-gray-50"}`}
              >
                {/* Champion badge */}
                <div className="w-5 flex-none flex items-center justify-center">
                  {p.is_champion
                    ? <Trophy className="w-3.5 h-3.5 text-amber-500" />
                    : <span className="text-[10px] font-mono text-gray-300">{i + 1}</span>}
                </div>

                {/* Name */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[12px] font-semibold text-gray-800 truncate">{prettyName(p.name)}</span>
                    {p.error && <AlertCircle className="w-3 h-3 text-red-400 flex-none" />}
                  </div>
                  {p.f1 != null && (
                    <div className="mt-1 h-1 w-full bg-gray-100 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gray-400 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${f1Pct}%` }}
                        transition={{ duration: 0.5, delay: i * 0.02 }}
                      />
                    </div>
                  )}
                </div>

                {/* Label pill */}
                <span className={`flex-none px-2 py-0.5 rounded-lg text-[10px] font-bold border ${lc2.bg} ${lc2.text} border-current/20`}>
                  {SEVERITY[p.label] ?? p.label}
                </span>

                {/* Confidence bar */}
                <div className="flex-none w-16 space-y-0.5">
                  {p.confidence != null ? (
                    <>
                      <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                        <motion.div
                          className={`h-full rounded-full ${lc2.dot}`}
                          initial={{ width: 0 }}
                          animate={{ width: `${confPct}%` }}
                          transition={{ duration: 0.5, delay: i * 0.02 }}
                        />
                      </div>
                      <div className="text-right font-mono text-[10px] text-gray-500">
                        {confPct.toFixed(0)}%
                      </div>
                    </>
                  ) : (
                    <span className="text-[10px] text-gray-300">no prob</span>
                  )}
                </div>

                {/* F1 */}
                {p.f1 != null && (
                  <div className="flex-none w-10 text-right font-mono text-[10px] text-gray-400">
                    F1 {(p.f1 * 100).toFixed(0)}
                  </div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* Show more / less */}
      {predictions.length > 8 && (
        <div className="px-5 py-3 border-t border-gray-100">
          <button
            onClick={() => setShowAll(s => !s)}
            className="w-full flex items-center justify-center gap-1.5 text-xs font-semibold text-gray-500 hover:text-gray-800 transition-colors py-1"
          >
            {showAll
              ? <><ChevronUp className="w-3.5 h-3.5" /> Show fewer models</>
              : <><ChevronDown className="w-3.5 h-3.5" /> Show all {predictions.length} models</>}
          </button>
        </div>
      )}
    </div>
  );
}
