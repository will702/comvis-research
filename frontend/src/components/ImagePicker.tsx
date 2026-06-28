import { useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, CheckCircle2 } from "lucide-react";
import type { DemoImage } from "../lib/api";

const LABEL_COLORS: Record<string, string> = {
  acne1: "bg-emerald-100 text-emerald-700 border-emerald-200",
  acne2: "bg-amber-100  text-amber-700  border-amber-200",
  acne3: "bg-rose-100   text-rose-700   border-rose-200",
};
const LABEL_DOTS: Record<string, string> = {
  acne1: "bg-emerald-500",
  acne2: "bg-amber-500",
  acne3: "bg-rose-500",
};

interface Props {
  demos: DemoImage[];
  selectedDemoId: string | null;
  onSelectDemo: (id: string) => void;
  onUpload: (file: File) => void;
  loading: boolean;
}

export default function ImagePicker({ demos, selectedDemoId, onSelectDemo, onUpload, loading }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploadName, setUploadName] = useState<string | null>(null);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) {
      setUploadName(f.name);
      onUpload(f);
    }
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) { setUploadName(f.name); onUpload(f); }
  }

  // Group demos by label
  const grouped: Record<string, DemoImage[]> = {};
  for (const d of demos) {
    grouped[d.label] = grouped[d.label] || [];
    grouped[d.label].push(d);
  }

  const severityLabel: Record<string, string> = {
    acne1: "Mild (Grade I)",
    acne2: "Moderate (Grade II)",
    acne3: "Severe (Grade III)",
  };

  return (
    <div className="space-y-6">
      {/* Upload zone */}
      <div>
        <h3 className="font-display font-semibold text-sm text-gray-500 uppercase tracking-widest mb-3">
          Upload Your Photo
        </h3>
        <motion.div
          className={`relative rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 overflow-hidden
            ${dragging ? "border-violet-400 bg-violet-50" : "border-gray-200 bg-gray-50 hover:border-violet-300 hover:bg-violet-50/40"}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          whileHover={{ scale: 1.005 }}
          whileTap={{ scale: 0.998 }}
        >
          <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />
          <div className="py-8 px-6 flex flex-col items-center gap-3 text-center">
            <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-colors
              ${dragging ? "bg-violet-200" : "bg-white shadow-sm border border-gray-100"}`}>
              <Upload className={`w-5 h-5 ${dragging ? "text-violet-600" : "text-gray-400"}`} />
            </div>
            {uploadName ? (
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-sm font-medium text-violet-700">
                  <CheckCircle2 className="w-4 h-4" />
                  {uploadName}
                </div>
                <p className="text-xs text-gray-400">Click to change</p>
              </div>
            ) : (
              <div className="space-y-1">
                <p className="text-sm font-semibold text-gray-700">Drag & drop or click to upload</p>
                <p className="text-xs text-gray-400">JPG, PNG — processed in memory, never stored</p>
              </div>
            )}
          </div>
          <AnimatePresence>
            {dragging && (
              <motion.div
                className="absolute inset-0 bg-violet-400/10 flex items-center justify-center"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              >
                <p className="text-violet-600 font-semibold text-lg">Drop it!</p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Demo gallery */}
      <div>
        <h3 className="font-display font-semibold text-sm text-gray-500 uppercase tracking-widest mb-3">
          Or Pick a Demo Image
        </h3>
        <div className="space-y-4">
          {Object.entries(grouped).sort().map(([label, imgs]) => (
            <div key={label}>
              <div className="flex items-center gap-2 mb-2">
                <span className={`w-2 h-2 rounded-full ${LABEL_DOTS[label] ?? "bg-gray-400"}`} />
                <span className="text-xs font-semibold text-gray-600">
                  {severityLabel[label] ?? label}
                </span>
              </div>
              <div className="grid grid-cols-4 gap-2">
                {imgs.map(d => (
                  <motion.button
                    key={d.id}
                    onClick={() => onSelectDemo(d.id)}
                    className={`relative rounded-xl overflow-hidden aspect-square ring-2 ring-offset-1 transition-all
                      ${selectedDemoId === d.id
                        ? "ring-violet-500 shadow-lg shadow-violet-200"
                        : "ring-transparent hover:ring-violet-300"}`}
                    whileHover={{ scale: 1.06 }}
                    whileTap={{ scale: 0.96 }}
                    disabled={loading}
                  >
                    <img
                      src={`data:image/png;base64,${d.thumb}`}
                      alt={d.label}
                      className="w-full h-full object-cover"
                    />
                    {selectedDemoId === d.id && (
                      <motion.div
                        className="absolute inset-0 bg-violet-500/20 flex items-center justify-center"
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      >
                        <CheckCircle2 className="w-5 h-5 text-white drop-shadow" />
                      </motion.div>
                    )}
                    <div className={`absolute bottom-1 left-1 px-1.5 py-0.5 rounded-md text-[10px] font-bold border ${LABEL_COLORS[d.label] ?? ""}`}>
                      {d.label}
                    </div>
                  </motion.button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
