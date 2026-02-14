import { useEffect, useMemo, useRef, useState } from "react";
// eslint-disable-next-line no-unused-vars
import { motion } from "motion/react";

const DEFAULT_CHARS =
  "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
const ULTRA_CHARS =
  "@$B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";

const container = {
  hidden: { opacity: 0, y: 22 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, ease: "easeOut", staggerChildren: 0.06 },
  },
};

const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" } },
};

export default function TextArt({ lang = "EN" }) {
  const isEN = lang === "EN";
  const [cols, setCols] = useState(120);
  const [chars, setChars] = useState(DEFAULT_CHARS);
  const [invert, setInvert] = useState(false);
  const [threshold, setThreshold] = useState(0);
  const [contrast, setContrast] = useState(1.4);
  const [fontSize, setFontSize] = useState(6);
  const [lineHeight, setLineHeight] = useState(0.8);
  const [edgeBoost, setEdgeBoost] = useState(0.25);
  const [imageSrc, setImageSrc] = useState("/775.png");
  const [ascii, setAscii] = useState("Loading /775.png...");
  const objectUrlRef = useRef("");

  const charRamp = useMemo(() => {
    const trimmed = chars.replace(/\s+/g, " ");
    return trimmed.length >= 2 ? trimmed : DEFAULT_CHARS;
  }, [chars]);

  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.src = imageSrc;
    img.crossOrigin = "anonymous";

    img.onload = () => {
      if (cancelled) return;
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      if (!context) return;

      const aspect = img.height / img.width;
      const width = Math.max(20, cols);
      const height = Math.max(10, Math.round(width * aspect * 0.5));
      canvas.width = width;
      canvas.height = height;

      context.drawImage(img, 0, 0, width, height);
      const { data } = context.getImageData(0, 0, width, height);
      const ramp = invert ? charRamp.split("").reverse() : charRamp.split("");
      const rampMax = ramp.length - 1;

      let output = "";
      for (let y = 0; y < height; y += 1) {
        let row = "";
        for (let x = 0; x < width; x += 1) {
          const idx = (y * width + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          const alpha = data[idx + 3] / 255;

          let luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) * alpha;
          if (contrast !== 1) {
            luminance = Math.min(255, Math.max(0, (luminance - 128) * contrast + 128));
          }
          if (threshold > 0) {
            luminance = luminance > threshold ? 255 : 0;
          }
          if (edgeBoost > 0) {
            const nx = Math.min(width - 1, x + 1);
            const ny = Math.min(height - 1, y + 1);
            const nidx = (ny * width + nx) * 4;
            const nr = data[nidx];
            const ng = data[nidx + 1];
            const nb = data[nidx + 2];
            const nlum = (0.2126 * nr + 0.7152 * ng + 0.0722 * nb) * alpha;
            const edge = Math.min(255, Math.max(0, Math.abs(luminance - nlum) * 2));
            luminance = Math.min(255, Math.max(0, luminance + edge * edgeBoost));
          }

          const charIndex = Math.round((luminance / 255) * rampMax);
          row += ramp[charIndex];
        }
        output += `${row}\n`;
      }
      setAscii(output);
    };

    img.onerror = () => {
      if (!cancelled) setAscii("Failed to load image");
    };

    return () => {
      cancelled = true;
    };
  }, [cols, charRamp, invert, threshold, contrast, edgeBoost, imageSrc]);

  useEffect(
    () => () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    []
  );

  const resetAll = () => {
    setImageSrc("/775.png");
    setCols(120);
    setChars(DEFAULT_CHARS);
    setInvert(false);
    setThreshold(0);
    setContrast(1.4);
    setEdgeBoost(0.25);
    setFontSize(6);
    setLineHeight(0.8);
  };

  return (
    <motion.main
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="relative min-h-screen overflow-hidden bg-[#050a1c] font-[Sarabun] text-slate-100"
    >
      <div className="pointer-events-none absolute inset-0">
        <motion.div
          animate={{ x: [0, 24, -14, 0], y: [0, -20, 18, 0] }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
          className="absolute -top-16 left-[18%] h-80 w-80 rounded-full bg-cyan-500/20 blur-[130px]"
        />
        <motion.div
          animate={{ x: [0, -26, 10, 0], y: [0, 16, -20, 0] }}
          transition={{ duration: 16, repeat: Infinity, ease: "easeInOut" }}
          className="absolute bottom-[-80px] right-[8%] h-[26rem] w-[26rem] rounded-full bg-emerald-400/15 blur-[150px]"
        />
      </div>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="relative mx-auto grid max-w-7xl gap-6 px-4 pb-10 pt-20 sm:px-8 lg:grid-cols-[1fr_1.2fr]"
      >
        <motion.section
          variants={item}
          className="rounded-3xl border border-cyan-300/20 bg-[#0b1734]/85 p-5 shadow-[0_20px_80px_rgba(0,0,0,0.45)] backdrop-blur-xl"
        >
          <motion.div variants={item}>
            <p className="text-[11px] uppercase tracking-[0.35em] text-cyan-300">
              {isEN ? "Image to ASCII" : "แปลงภาพเป็น ASCII"}
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">
              {isEN ? "Text Art Studio" : "สตูดิโอภาพจากตัวอักษร"}
            </h1>
            <p className="mt-2 text-sm text-slate-300">
              {isEN
                ? "Upload image, tune conversion settings, and export dense ASCII detail."
                : "อัปโหลดรูป ปรับค่าการแปลง และสร้าง ASCII แบบละเอียดได้ทันที"}
            </p>
          </motion.div>

          <motion.div variants={item} className="mt-5">
            <label className="text-[11px] uppercase tracking-[0.3em] text-slate-400">
              {isEN ? "Upload image" : "อัปโหลดรูป"}
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
                const next = URL.createObjectURL(file);
                objectUrlRef.current = next;
                setImageSrc(next);
              }}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0f1831] px-3 py-2 text-xs text-slate-300 file:mr-3 file:rounded-lg file:border-0 file:bg-cyan-500/80 file:px-3 file:py-1 file:text-white"
            />
          </motion.div>

          <motion.div variants={item} className="mt-5">
            <label className="text-[11px] uppercase tracking-[0.3em] text-slate-400">
              {isEN ? "Character ramp" : "ชุดตัวอักษร"}
            </label>
            <input
              type="text"
              value={chars}
              onChange={(event) => setChars(event.target.value)}
              placeholder={DEFAULT_CHARS}
              className="mt-2 w-full rounded-2xl border border-white/10 bg-[#0f172a] px-3 py-3 text-sm text-slate-100 placeholder:text-slate-500"
            />
          </motion.div>

          <motion.div variants={item} className="mt-5 grid gap-4 sm:grid-cols-2">
            <label className="text-[11px] uppercase tracking-[0.3em] text-slate-400">
              {isEN ? "Columns" : "จำนวนคอลัมน์"}: {cols}
              <input
                type="range"
                min="80"
                max="260"
                value={cols}
                onChange={(event) => setCols(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-400"
              />
            </label>
            <label className="text-[11px] uppercase tracking-[0.3em] text-slate-400">
              {isEN ? "Contrast" : "คอนทราสต์"}: {contrast.toFixed(1)}x
              <input
                type="range"
                min="0.6"
                max="2.5"
                step="0.1"
                value={contrast}
                onChange={(event) => setContrast(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-400"
              />
            </label>
            <label className="text-[11px] uppercase tracking-[0.3em] text-slate-400">
              {isEN ? "Threshold" : "ค่าแยกแสง"}: {threshold}
              <input
                type="range"
                min="0"
                max="255"
                value={threshold}
                onChange={(event) => setThreshold(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-400"
              />
            </label>
            <label className="text-[11px] uppercase tracking-[0.3em] text-slate-400">
              {isEN ? "Edge Boost" : "เพิ่มความคมขอบ"}: {edgeBoost.toFixed(2)}
              <input
                type="range"
                min="0"
                max="0.6"
                step="0.05"
                value={edgeBoost}
                onChange={(event) => setEdgeBoost(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-400"
              />
            </label>
            <label className="text-[11px] uppercase tracking-[0.3em] text-slate-400">
              {isEN ? "Font Size" : "ขนาดตัวอักษร"}: {fontSize}px
              <input
                type="range"
                min="3"
                max="10"
                value={fontSize}
                onChange={(event) => setFontSize(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-400"
              />
            </label>
            <label className="text-[11px] uppercase tracking-[0.3em] text-slate-400">
              {isEN ? "Line Height" : "ระยะบรรทัด"}: {lineHeight}
              <input
                type="range"
                min="0.6"
                max="1.1"
                step="0.05"
                value={lineHeight}
                onChange={(event) => setLineHeight(Number(event.target.value))}
                className="mt-2 w-full accent-cyan-400"
              />
            </label>
          </motion.div>

          <motion.div variants={item} className="mt-5 flex items-center gap-2 text-sm text-slate-300">
            <input
              id="invert"
              type="checkbox"
              checked={invert}
              onChange={(event) => setInvert(event.target.checked)}
              className="h-4 w-4 rounded border-white/20 bg-[#0f172a] accent-cyan-400"
            />
            <label htmlFor="invert">{isEN ? "Invert ramp" : "กลับค่าชุดตัวอักษร"}</label>
          </motion.div>

          <motion.div variants={item} className="mt-5 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => {
                setCols(220);
                setChars(ULTRA_CHARS);
                setInvert(false);
                setThreshold(0);
                setContrast(1.8);
                setEdgeBoost(0.35);
                setFontSize(4);
                setLineHeight(0.75);
              }}
              className="rounded-full bg-cyan-500/80 px-4 py-2 text-xs uppercase tracking-[0.25em] text-white transition hover:bg-cyan-400"
            >
              {isEN ? "HQ Preset" : "พรีเซ็ตละเอียด"}
            </button>
            <button
              type="button"
              onClick={resetAll}
              className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs uppercase tracking-[0.25em] text-slate-200 transition hover:border-cyan-300/60 hover:text-cyan-100"
            >
              {isEN ? "Reset" : "รีเซ็ต"}
            </button>
          </motion.div>
        </motion.section>

        <motion.section
          variants={item}
          className="rounded-3xl border border-emerald-300/20 bg-[#0a1430]/85 p-5 shadow-[0_20px_80px_rgba(0,0,0,0.45)] backdrop-blur-xl"
        >
          <div className="mb-4 flex items-center justify-between">
            <p className="text-[11px] uppercase tracking-[0.35em] text-emerald-300">
              {isEN ? "Preview" : "พรีวิว"}
            </p>
            <span className="text-xs text-slate-400">{ascii.length.toLocaleString()} chars</span>
          </div>

          <motion.div
            key={`${cols}-${contrast}-${threshold}-${fontSize}-${lineHeight}-${edgeBoost}-${invert}-${charRamp}`}
            initial={{ opacity: 0, scale: 0.985 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="h-[70vh] overflow-auto rounded-2xl border border-white/10 bg-[#050915] p-4"
          >
            <pre
              className="whitespace-pre font-mono text-slate-100"
              style={{ fontSize: `${fontSize}px`, lineHeight }}
            >
              {ascii}
            </pre>
          </motion.div>
          <p className="mt-3 text-xs text-slate-400">
            {isEN
              ? "Higher columns plus denser character ramp gives more detail."
              : "เพิ่มจำนวนคอลัมน์และเพิ่มความถี่ชุดตัวอักษร จะช่วยให้ภาพละเอียดขึ้น"}
          </p>
        </motion.section>
      </motion.div>
    </motion.main>
  );
}
