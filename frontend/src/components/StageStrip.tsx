import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, Info } from "lucide-react";
import { b64img } from "../lib/api";
import type { Trace } from "../lib/api";

interface StageConfig {
  id: string;
  num: number;
  title: string;
  subtitle: string;
  color: string;        // tailwind color name
  accent: string;       // bg class for badge
  textAccent: string;
  borderAccent: string;
  images: (keyof Trace)[];
  labels?: string[];
  explain: string;
  metaFn?: (t: Trace) => React.ReactNode;
}

const STAGES: StageConfig[] = [
  {
    id: "resize", num: 1, title: "Resize", subtitle: "512 × 512 pixels",
    color: "slate", accent: "bg-slate-800", textAccent: "text-slate-700", borderAccent: "border-slate-200",
    images: ["img_resized"],
    explain: "Every image is scaled to a fixed 512×512 grid. Fixed size lets all downstream operations use hardcoded kernel sizes and cascade min-sizes that are calibrated for this resolution.",
  },
  {
    id: "gray", num: 2, title: "Grayscale", subtitle: "Luminance channel",
    color: "gray", accent: "bg-gray-700", textAccent: "text-gray-700", borderAccent: "border-gray-200",
    images: ["img_gray"],
    explain: "BGR → grayscale collapses three channels to one luminance value per pixel. This is the input for CLAHE enhancement and Haar cascade face detection — both work on intensity, not color.",
  },
  {
    id: "clahe", num: 3, title: "CLAHE", subtitle: "Adaptive histogram equalisation",
    color: "amber", accent: "bg-amber-500", textAccent: "text-amber-700", borderAccent: "border-amber-200",
    images: ["img_gray", "img_clahe"],
    labels: ["Before", "After CLAHE"],
    explain: "CLAHE (Contrast Limited Adaptive Histogram Equalisation) enhances local contrast tile-by-tile. Unlike global equalisation it doesn't over-amplify uniform regions. The clip limit caps noise amplification.",
    metaFn: (t) => (
      <div className="flex gap-3 flex-wrap">
        <Chip label="Clip limit" value={t.thr_cr ? "→" : "3.0"} color="amber" />
        <Chip label="Tile size" value="8×8" color="amber" />
      </div>
    ),
  },
  {
    id: "face", num: 4, title: "Face Detection", subtitle: "Haar cascade chain",
    color: "green", accent: "bg-green-600", textAccent: "text-green-700", borderAccent: "border-green-200",
    images: ["img_face_box"],
    explain: "Four cascade attempts relax parameters progressively (scale factor, min-neighbours, min-size) and switch to the alt2 cascade if the default fails. Acne-texture skin fools single-pass detectors.",
    metaFn: (t) => (
      <div className="flex gap-3 flex-wrap items-center">
        {t.face_detected
          ? <><Chip label="Face found" value={`attempt #${t.cascade_attempt + 1}`} color="green" />
              <Chip label="Box" value={t.face_box ? `${t.face_box[2]}×${t.face_box[3]}px` : "—"} color="green" /></>
          : <Chip label="No face" value="using full 512×512" color="gray" />}
      </div>
    ),
  },
  {
    id: "roi", num: 5, title: "ROI Mask", subtitle: "Eyes · Nose · Lips excluded",
    color: "red", accent: "bg-red-500", textAccent: "text-red-700", borderAccent: "border-red-200",
    images: ["img_roi_mask"],
    explain: "Eyes, nose, and mouth produce red signals (blood vessels, lips) that aren't acne. Haar cascades detect these sub-regions; their bounding boxes are zeroed in the ROI mask so they never feed lesion detection.",
    metaFn: (t) => (
      <div className="flex gap-3 flex-wrap">
        {t.roi_regions.map((r, i) => <Chip key={i} label={r.type} value="excluded" color="red" />)}
        {t.roi_regions.length === 0 && <Chip label="No face → no exclusions" value="full mask used" color="gray" />}
      </div>
    ),
  },
  {
    id: "skin", num: 6, title: "Skin Segmentation", subtitle: "YCrCb colour range",
    color: "orange", accent: "bg-orange-500", textAccent: "text-orange-700", borderAccent: "border-orange-200",
    images: ["img_skin_raw", "img_skin_mask"],
    labels: ["YCrCb mask only", "Skin ∩ ROI (final)"],
    explain: "YCrCb separates luma from chroma. Skin pixels fall in a narrow Cr/Cb band regardless of illumination. MORPH_CLOSE fills holes; MORPH_OPEN removes noise. The final mask is the intersection with the ROI mask.",
    metaFn: (t) => (
      <div className="flex gap-3 flex-wrap">
        <Chip label="Skin pixels" value={t.skin_pixel_count.toLocaleString()} color="orange" />
      </div>
    ),
  },
  {
    id: "thresh", num: 7, title: "Adaptive Thresholds", subtitle: "Red mask from Cr + a* channels",
    color: "rose", accent: "bg-rose-500", textAccent: "text-rose-700", borderAccent: "border-rose-200",
    images: ["img_red_mask"],
    explain: "Active acne is elevated in both YCrCb Cr (chroma-red) AND LAB a* (red-green axis). The thresholds adapt to each image's skin statistics: thr = max(floor, μ + k·σ). Requiring BOTH channels above threshold eliminates brown scars and lip redness.",
    metaFn: (t) => (
      <div className="flex gap-3 flex-wrap">
        <Chip label="thr_cr" value={t.thr_cr.toFixed(1)} color="rose" />
        <Chip label="thr_a*" value={t.thr_a.toFixed(1)} color="rose" />
        <Chip label="μCr" value={t.mu_cr.toFixed(1)} color="rose" />
        <Chip label="μa*" value={t.mu_a.toFixed(1)} color="rose" />
      </div>
    ),
  },
  {
    id: "blobs", num: 8, title: "Lesion Detection", subtitle: "Connected components + shape filter",
    color: "pink", accent: "bg-pink-500", textAccent: "text-pink-700", borderAccent: "border-pink-200",
    images: ["img_blobs"],
    explain: "Connected components label each red blob. Five filters reject non-lesions: min size (too small = noise), max size (too large = diffuse redness), aspect ratio (too elongated = wrinkles), fill ratio (too hollow), and local a* contrast (not significantly redder than surrounding skin).",
    metaFn: (t) => (
      <div className="flex gap-3 flex-wrap">
        <Chip label="Kept" value={String(t.kept_blobs)} color="pink" />
        <Chip label="Rejected" value={String(t.rejected_blobs)} color="gray" />
        {t.blobs.filter(b => !b.kept).reduce<Record<string,number>>((acc, b) => {
          const r = b.reject_reason ?? "?"; acc[r] = (acc[r] || 0) + 1; return acc;
        }, {}) && null}
      </div>
    ),
  },
  {
    id: "structural", num: 9, title: "Structural Features", subtitle: "8 lesion statistics",
    color: "blue", accent: "bg-blue-600", textAccent: "text-blue-700", borderAccent: "border-blue-200",
    images: [],
    explain: "Eight scalar statistics summarise detected lesion blobs: count, total area, intensity mean/std (LAB a* redness), max area, area std-dev, lesion density (count / skin px), and mean circularity. These form dims 0–7 of the 42-dim vector.",
    metaFn: (t) => {
      const v = t.feature_vector;
      const rows = [
        ["lesion_count", v[0].toFixed(0)],
        ["total_area", v[1].toFixed(0)+"px²"],
        ["intensity_mean", v[2].toFixed(2)],
        ["lesion_density", v[6].toExponential(2)],
        ["mean_circularity", v[7].toFixed(3)],
      ];
      return (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {rows.map(([k, v]) => (
            <div key={k} className="flex justify-between items-center">
              <span className="font-mono text-[10px] text-blue-600">{k}</span>
              <span className="font-mono text-[10px] font-bold text-blue-900">{v}</span>
            </div>
          ))}
        </div>
      );
    },
  },
  {
    id: "lbp", num: 10, title: "Multi-scale LBP", subtitle: "Local Binary Patterns · R = 1, 2, 3",
    color: "indigo", accent: "bg-indigo-600", textAccent: "text-indigo-700", borderAccent: "border-indigo-200",
    images: ["img_lbp_r1", "img_lbp_r2", "img_lbp_r3"],
    labels: ["R=1 (fine)", "R=2 (medium)", "R=3 (coarse)"],
    explain: "LBP encodes local texture by comparing each pixel to its P=8 circular neighbours at radius R. The 'uniform' method counts transitions; the histogram over skin pixels captures surface roughness. Three radii capture texture at different scales → 27 dims (9 bins × 3 radii, last dependent bin dropped).",
    metaFn: (t) => (
      <div className="grid grid-cols-3 gap-2">
        {(["img_lbp_hist_r1","img_lbp_hist_r2","img_lbp_hist_r3"] as (keyof Trace)[]).map((k, i) => (
          t[k] ? <img key={k} src={b64img(t[k] as string)} alt={`LBP R=${i+1} hist`} className="w-full rounded border border-indigo-100" /> : null
        ))}
      </div>
    ),
  },
  {
    id: "glcm", num: 11, title: "GLCM Texture", subtitle: "Grey-Level Co-occurrence Matrix",
    color: "cyan", accent: "bg-cyan-600", textAccent: "text-cyan-700", borderAccent: "border-cyan-200",
    images: [],
    explain: "GLCM counts how often pixel intensity pairs co-occur at distance=1, across 4 angles (0°, 45°, 90°, 135°). Three properties are extracted: contrast (local variance), homogeneity (inverse contrast), and energy (orderliness). Dissimilarity was removed due to 0.95 correlation with contrast.",
    metaFn: (t) => (
      <div className="flex gap-4">
        {Object.entries(t.glcm).map(([k, v]) => (
          <div key={k} className="text-center">
            <div className="font-mono text-xs font-bold text-cyan-800">{(v as number).toFixed(4)}</div>
            <div className="text-[10px] text-cyan-600">{k}</div>
          </div>
        ))}
      </div>
    ),
  },
  {
    id: "redness", num: 12, title: "Global Redness", subtitle: "LAB a* and YCrCb Cr heatmaps",
    color: "red", accent: "bg-red-600", textAccent: "text-red-700", borderAccent: "border-red-200",
    images: ["img_a_heat", "img_cr_heat"],
    labels: ["LAB a* (red–green)", "YCrCb Cr (chroma-red)"],
    explain: "Global skin redness is measured independent of lesion detection. μ and σ of a* and Cr across all skin pixels capture the overall inflammatory state. In Grade I the whole face is neutral; in Grade III the whole face is inflamed — a signal the adaptive threshold was hiding. These 4 values form dims 38–41.",
    metaFn: (t) => (
      <div className="flex gap-3 flex-wrap">
        <Chip label="μ a*" value={t.mu_a.toFixed(2)} color="red" />
        <Chip label="σ a*" value={t.std_a.toFixed(2)} color="red" />
        <Chip label="μ Cr" value={t.mu_cr.toFixed(2)} color="red" />
        <Chip label="σ Cr" value={t.std_cr.toFixed(2)} color="red" />
      </div>
    ),
  },
];

// ── Chip helper ────────────────────────────────────────────────────────────────
const CHIP_COLORS: Record<string, string> = {
  amber:  "bg-amber-50  text-amber-800  border-amber-200",
  green:  "bg-green-50  text-green-800  border-green-200",
  red:    "bg-red-50    text-red-800    border-red-200",
  rose:   "bg-rose-50   text-rose-800   border-rose-200",
  orange: "bg-orange-50 text-orange-800 border-orange-200",
  pink:   "bg-pink-50   text-pink-800   border-pink-200",
  blue:   "bg-blue-50   text-blue-800   border-blue-200",
  indigo: "bg-indigo-50 text-indigo-800 border-indigo-200",
  cyan:   "bg-cyan-50   text-cyan-800   border-cyan-200",
  gray:   "bg-gray-50   text-gray-700   border-gray-200",
};

function Chip({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-lg text-[11px] border font-medium ${CHIP_COLORS[color] ?? CHIP_COLORS.gray}`}>
      <span className="opacity-60">{label}</span>
      <span className="font-mono font-bold">{value}</span>
    </span>
  );
}

// ── Stage card ─────────────────────────────────────────────────────────────────
function StageCard({ stage, trace, expanded, onToggle }: {
  stage: StageConfig;
  trace: Trace | null;
  expanded: boolean;
  onToggle: () => void;
}) {
  const imgs = stage.images.map(k => trace ? (trace[k] as string) : "");
  const hasImages = imgs.some(Boolean);

  return (
    <motion.div
      className={`rounded-2xl border overflow-hidden transition-shadow ${expanded ? "shadow-md" : "shadow-sm hover:shadow-md"} ${stage.borderAccent} bg-white`}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      {/* Header row */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-3.5 text-left hover:bg-gray-50/60 transition-colors"
      >
        <span className={`flex-none w-7 h-7 rounded-lg ${stage.accent} flex items-center justify-center text-white text-xs font-bold font-display`}>
          {stage.num}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2">
            <span className="font-display font-bold text-sm text-gray-900">{stage.title}</span>
            <span className={`text-[11px] font-medium ${stage.textAccent}`}>{stage.subtitle}</span>
          </div>
        </div>
        {!trace && <span className="text-[10px] text-gray-300 italic">run pipeline first</span>}
        <motion.div animate={{ rotate: expanded ? 90 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronRight className="w-4 h-4 text-gray-300" />
        </motion.div>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-4 border-t border-gray-100 pt-4">
              {/* Explain */}
              <p className="text-[13px] text-gray-600 leading-relaxed">{stage.explain}</p>

              {/* Images */}
              {trace && hasImages && (
                <div className={`grid gap-2 ${imgs.filter(Boolean).length > 1 ? "grid-cols-2 lg:grid-cols-3" : "grid-cols-1"}`}>
                  {imgs.map((src, i) => src ? (
                    <div key={i} className="space-y-1">
                      {stage.labels?.[i] && (
                        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">{stage.labels[i]}</p>
                      )}
                      <img
                        src={b64img(src)}
                        alt={`${stage.title} output ${i + 1}`}
                        className="w-full rounded-xl border border-gray-100 object-cover"
                      />
                    </div>
                  ) : null)}
                </div>
              )}

              {/* Meta / chips */}
              {trace && stage.metaFn && (
                <div className={`rounded-xl p-3 bg-gray-50 border border-gray-100`}>
                  {stage.metaFn(trace)}
                </div>
              )}

              {!trace && (
                <div className="flex items-center gap-2 py-3 text-gray-300">
                  <Info className="w-4 h-4" />
                  <span className="text-xs">Select an image above to see live output</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────
interface Props {
  trace: Trace | null;
}

export default function StageStrip({ trace }: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>(
    Object.fromEntries(STAGES.map(s => [s.id, true]))
  );

  function toggle(id: string) {
    setExpanded(e => ({ ...e, [id]: !e[id] }));
  }

  function expandAll() { setExpanded(Object.fromEntries(STAGES.map(s => [s.id, true]))); }
  function collapseAll() { setExpanded(Object.fromEntries(STAGES.map(s => [s.id, false]))); }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="font-display font-bold text-lg text-gray-900">Pipeline Stages</h2>
        <div className="flex gap-2">
          <button onClick={expandAll}  className="text-xs text-gray-400 hover:text-gray-700 transition-colors">Expand all</button>
          <span className="text-gray-200">·</span>
          <button onClick={collapseAll} className="text-xs text-gray-400 hover:text-gray-700 transition-colors">Collapse all</button>
        </div>
      </div>

      <div className="space-y-2">
        {STAGES.map(s => (
          <StageCard
            key={s.id}
            stage={s}
            trace={trace}
            expanded={expanded[s.id] ?? true}
            onToggle={() => toggle(s.id)}
          />
        ))}
      </div>
    </div>
  );
}
