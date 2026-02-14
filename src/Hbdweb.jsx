import { useEffect, useMemo, useRef, useState } from "react";
import { Volume2, VolumeX } from 'lucide-react';

export default function Hbdweb() {
  const [name, setName] = useState("สมปอง");
  const [message, setMessage] = useState(
  `Happy birthday to you, Happy birthday to you Happy birthday Happy birthday Happy birthday to you.
   Happy birthday to you, Happy birthday to you Happy birthday Happy birthday Happy birthday to you.`
);
  const [isConfetti, setIsConfetti] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [bgType, setBgType] = useState("none");
  const [bgUrl, setBgUrl] = useState("");
  const videoRef = useRef(null);
  const [isMuted, setIsMuted] = useState(true);
  const [crop, setCrop] = useState({ x: 10, y: 10, w: 80, h: 80 });
  const cropStateRef = useRef({ mode: null, startX: 0, startY: 0, start: null, rect: null });
  const cropBoxRef = useRef(null);
  const [heroImage, setHeroImage] = useState("");

  const headline = useMemo(() => `Happy Birthday, ${name}!`, [name]);
  const overlayClass = bgUrl ? "bg-white/30" : "bg-white";

  useEffect(() => {
    if (!isFullscreen) return;
    const handler = (event) => {
      if (event.key === "Escape") setIsFullscreen(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isFullscreen]);


  useEffect(() => {
    const media = bgType === "video" && bgUrl ? videoRef.current : null;
    if (!media) return;
    media.muted = isMuted;
  }, [bgType, bgUrl, isMuted]);

  useEffect(() => {
    const handleKey = (event) => {
      if (event.key?.toLowerCase() === "m") {
        setIsMuted((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  useEffect(() => {
    return () => {
      if (bgUrl) URL.revokeObjectURL(bgUrl);
    };
  }, [bgUrl]);

  const handleBgFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (bgUrl) URL.revokeObjectURL(bgUrl);
    setBgUrl(URL.createObjectURL(file));
    setBgType(file.type.startsWith("video") ? "video" : "image");
    setCrop({ x: 10, y: 10, w: 80, h: 80 });
  };

  const handleHeroImage = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setHeroImage(reader.result);
      }
    };
    reader.readAsDataURL(file);
  };

  const startCropDrag = (event) => {
    if (!cropBoxRef.current) return;
    const rect = cropBoxRef.current.getBoundingClientRect();
    cropStateRef.current = {
      mode: "move",
      startX: event.clientX,
      startY: event.clientY,
      start: { ...crop },
      rect,
    };
    event.preventDefault();
  };

  const startCropResize = (mode) => (event) => {
    if (!cropBoxRef.current) return;
    const rect = cropBoxRef.current.getBoundingClientRect();
    cropStateRef.current = {
      mode,
      startX: event.clientX,
      startY: event.clientY,
      start: { ...crop },
      rect,
    };
    event.preventDefault();
  };

  useEffect(() => {
    const handleMove = (event) => {
      const state = cropStateRef.current;
      if (!state.mode || !state.rect) return;
      const dx = ((event.clientX - state.startX) / state.rect.width) * 100;
      const dy = ((event.clientY - state.startY) / state.rect.height) * 100;
      const minW = 20;
      const minH = 20;
      if (state.mode === "move") {
        const nextX = Math.min(100 - state.start.w, Math.max(0, state.start.x + dx));
        const nextY = Math.min(100 - state.start.h, Math.max(0, state.start.y + dy));
        setCrop((prev) => ({ ...prev, x: nextX, y: nextY }));
        return;
      }
      let next = { ...state.start };
      if (state.mode.includes("e")) {
        next.w = Math.min(100 - next.x, Math.max(minW, state.start.w + dx));
      }
      if (state.mode.includes("s")) {
        next.h = Math.min(100 - next.y, Math.max(minH, state.start.h + dy));
      }
      if (state.mode.includes("w")) {
        const newX = Math.max(0, Math.min(state.start.x + dx, state.start.x + state.start.w - minW));
        next.w = state.start.w + (state.start.x - newX);
        next.x = newX;
      }
      if (state.mode.includes("n")) {
        const newY = Math.max(0, Math.min(state.start.y + dy, state.start.y + state.start.h - minH));
        next.h = state.start.h + (state.start.y - newY);
        next.y = newY;
      }
      setCrop(next);
    };

    const handleUp = () => {
      cropStateRef.current = { mode: null, startX: 0, startY: 0, start: null, rect: null };
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };
  }, []);

  return (
    <main
      className={`relative min-h-screen overflow-hidden text-slate-900 font-['Baloo_2',_Poppins,system-ui]`}
    >
      <style>{`
        @keyframes confettiFall {
          0% { transform: translateY(-10vh) rotate(0deg); opacity: 0; }
          10% { opacity: 1; }
          100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
        }
        @keyframes stickerFall {
          0% { transform: translateY(-12vh) rotate(var(--rot, 0deg)); opacity: 0; }
          10% { opacity: 1; }
          100% { transform: translateY(110vh) rotate(calc(var(--rot, 0deg) + 6deg)); opacity: 0; }
        }
        @keyframes stickerDrift {
          0% { transform: translate(-6vw, -12vh) rotate(var(--rot, 0deg)); opacity: 0; }
          10% { opacity: 1; }
          50% { transform: translate(2vw, 40vh) rotate(calc(var(--rot, 0deg) + 6deg)); }
          100% { transform: translate(-4vw, 110vh) rotate(calc(var(--rot, 0deg) - 6deg)); opacity: 0; }
        }
        @keyframes stickerDriftAlt {
          0% { transform: translate(6vw, -12vh) rotate(var(--rot, 0deg)); opacity: 0; }
          10% { opacity: 1; }
          50% { transform: translate(-2vw, 45vh) rotate(calc(var(--rot, 0deg) - 6deg)); }
          100% { transform: translate(4vw, 110vh) rotate(calc(var(--rot, 0deg) + 6deg)); opacity: 0; }
        }
        .sticker-fall img {
          position: absolute;
          width: 80px;
          height: 80px;
          object-fit: contain;
          animation: stickerFall 1s linear infinite;
          filter: drop-shadow(0 6px 14px rgba(15, 23, 42, 0.2));
        }
        @keyframes floaty {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-14px); }
        }
        @keyframes sway {
          0%, 100% { transform: rotate(-2deg); }
          50% { transform: rotate(2deg); }
        }
        .confetti span {
          position: absolute;
          width: 10px;
          height: 16px;
          border-radius: 8px;
          animation: confettiFall 8s linear infinite;
        }
        .bunting span {
          display: inline-block;
          width: 32px;
          height: 24px;
          clip-path: polygon(50% 100%, 0 0, 100% 0);
          margin: 0 5px;
        }
        .floaty { animation: floaty 4s ease-in-out infinite; }
        .balloon {
          position: absolute;
          width: 120px;
          height: 150px;
          border-radius: 50% 50% 45% 45%;
          box-shadow: inset -12px -18px 0 rgba(0,0,0,0.06);
          animation: floaty 5s ease-in-out infinite;
        }
        .balloon .string {
          position: absolute;
          left: 50%;
          bottom: -70px;
          width: 2px;
          height: 70px;
          background: rgba(15, 23, 42, 0.25);
          transform: translateX(-50%);
        }
        .cake {
          position: relative;
          width: 220px;
          height: 160px;
          border-radius: 24px;
          background: #fde68a;
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.15);
        }
        .cake::before {
          content: "";
          position: absolute;
          top: -30px;
          left: 20px;
          right: 20px;
          height: 50px;
          border-radius: 20px;
          background: #fca5a5;
          box-shadow: inset 0 -8px 0 rgba(255, 255, 255, 0.35);
        }
        .cake::after {
          content: "";
          position: absolute;
          top: -52px;
          left: 90px;
          width: 40px;
          height: 30px;
          border-radius: 20px 20px 10px 10px;
          background: #fb7185;
          box-shadow: 0 -10px 0 #fcd34d;
        }
        .gift {
          position: relative;
          width: 120px;
          height: 90px;
          border-radius: 16px;
          background: #60a5fa;
          box-shadow: 0 14px 30px rgba(15, 23, 42, 0.15);
        }
        .gift::before {
          content: "";
          position: absolute;
          left: 50%;
          top: 0;
          width: 18px;
          height: 100%;
          background: #facc15;
          transform: translateX(-50%);
        }
        .gift::after {
          content: "";
          position: absolute;
          top: -18px;
          left: 36px;
          width: 48px;
          height: 20px;
          border-radius: 999px;
          background: #f472b6;
        }
        .bg-cover {
          position: absolute;
          inset: 0;
          overflow: hidden;
        }
        .bg-tile {
          position: absolute;
          inset: 0;
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          grid-template-rows: repeat(3, 1fr);
          overflow: hidden;
        }
        .bg-tile video {
          width: 100%;
          height: 100%;
          object-fit: cover;
          opacity: 0.6;
        }
        .crop-frame {
          position: absolute;
          border: 2px solid rgba(14, 165, 233, 0.8);
          border-radius: 12px;
          box-shadow: 0 0 0 9999px rgba(15, 23, 42, 0.2);
        }
        .crop-handle {
          position: absolute;
          width: 12px;
          height: 12px;
          background: #0ea5e9;
          border-radius: 50%;
          border: 2px solid #ffffff;
        }
        .crop-handle.nw { top: -6px; left: -6px; cursor: nwse-resize; }
        .crop-handle.ne { top: -6px; right: -6px; cursor: nesw-resize; }
        .crop-handle.sw { bottom: -6px; left: -6px; cursor: nesw-resize; }
        .crop-handle.se { bottom: -6px; right: -6px; cursor: nwse-resize; }
        @keyframes shimmer {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes glitter {
          0%, 100% { filter: saturate(1.2) brightness(1); }
          50% { filter: saturate(1.5) brightness(1.15); }
        }
        @keyframes sparkle {
          0%, 100% { opacity: 0.4; transform: scale(0.9); }
          50% { opacity: 1; transform: scale(1.1); }
        }
        .headline-wow {
          position: relative;
          font-weight: 700;
          letter-spacing: 0.02em;
          background: linear-gradient(
            120deg,
            #e11d48,
            #f59e0b,
            #0ea5e9,
            #7c3aed,
            #db2777
          );
          background-size: 400% 400%;
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          -webkit-text-fill-color: transparent;
          filter: saturate(1.35) brightness(1.05);
          text-shadow:
            0 2px 0 rgba(255, 255, 255, 0.9),
            0 8px 18px rgba(15, 23, 42, 0.18),
            0 14px 28px rgba(15, 23, 42, 0.12),
            0 0 18px rgba(234, 88, 12, 0.35),
            0 0 28px rgba(14, 165, 233, 0.35);
          animation: shimmer 6s ease-in-out infinite, glitter 2.4s ease-in-out infinite;
        }
        .headline-shell {
          display: inline-block;
          padding: 10px 22px;
          border-radius: 999px;
          background: #ffffff;
          border: 2px solid #fde68a;
          box-shadow:
            0 10px 24px rgba(15, 23, 42, 0.08),
            0 0 0 6px rgba(254, 240, 138, 0.35);
        }
        .headline-wow span {
          display: inline-block;
          padding: 0 8px;
        }
        .headline-glow {
          text-shadow:
            0 0 10px rgba(56, 189, 248, 0.4),
            0 0 24px rgba(248, 113, 113, 0.45);
        }
        .sub-wow {
          position: relative;
          display: inline-block;
          padding: 8px 18px;
          border-radius: 999px;
          background: #fff1f2;
          color: #f43f5e;
          font-weight: 600;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          font-size: 11px;
        }
      `}</style>

      {!isFullscreen && (
        <>
          {bgUrl && bgType === "image" && (
            <img
              src={bgUrl}
              alt=""
              className="pointer-events-none absolute inset-0 -z-10 h-full w-full object-cover opacity-60"
            />
          )}
          {bgUrl && bgType === "video" && (
            <div className="bg-tile pointer-events-none -z-10">
              {Array.from({ length: 9 }).map((_, i) => (
                <video
                  key={`tile-${i}`}
                  ref={i === 0 ? videoRef : null}
                  src={bgUrl}
                  autoPlay
                  loop
                  muted={i === 0 ? isMuted : true}
                  playsInline
                  style={{
                    transformOrigin: "top left",
                    transform: `translate(${-crop.x}%, ${-crop.y}%) scale(${100 / crop.w}, ${100 / crop.h})`,
                  }}
                />
              ))}
            </div>
          )}
          <div className={`pointer-events-none absolute inset-0 -z-0 ${overlayClass}`} />
        </>
      )}

      {isConfetti && (
        <div className="sticker-fall pointer-events-none absolute inset-0">
          {Array.from({ length: 18 }).map((_, i) => (
            <img
              key={`st-${i}`}
              src="/775.png"
              alt=""
              style={{
                left: `${(i * 7) % 100}vw`,
                top: `${(i * 11) % 20}vh`,
                animationDelay: `${(i % 9) * 0.4}s`,
                animationDuration: `${7 + (i % 6)}s`,
                animationName:
                  i % 3 === 0
                    ? "stickerDrift"
                    : i % 3 === 1
                    ? "stickerDriftAlt"
                    : "stickerFall",
                "--rot": `${(i % 5) * 3 - 6}deg`,
              }}
            />
          ))}
        </div>
      )}

      <div className="pointer-events-none absolute inset-0">
        <div className="balloon left-6 top-16 bg-[#fca5a5] sm:left-12">
          <span className="string" />
        </div>
        <div className="balloon right-6 top-24 bg-[#60a5fa] sm:right-12">
          <span className="string" />
        </div>
        <div className="balloon left-24 bottom-16 h-[120px] w-[90px] bg-[#facc15] sm:left-32">
          <span className="string" />
        </div>
      </div>

      <div className="relative mx-auto flex min-h-screen max-w-6xl flex-col px-6 pb-16 pt-20 sm:px-10">
        <div className="bunting mx-auto mb-6 flex flex-wrap justify-center">
          {Array.from({ length: 20 }).map((_, i) => (
            <span
              key={`b-${i}`}
              style={{
                background:
                  i % 4 === 0
                    ? "#f87171"
                    : i % 4 === 1
                    ? "#38bdf8"
                    : i % 4 === 2
                    ? "#a78bfa"
                    : "#facc15",
              }}
            />
          ))}
        </div>

        <header className="mb-10 rounded-[32px] border border-[#f1f5f9] bg-white p-10 text-center shadow-[0_30px_60px_rgba(15,23,42,0.12)]">
          <p className="text-xs uppercase tracking-[0.5em] text-slate-400">
            Celebrate
          </p>
          <div className="mt-4 flex flex-col items-center gap-3">
            <span className="sub-wow">Celebrate</span>
            {heroImage && (
              <div className="mb-4 flex justify-center">
                <img
                  src={heroImage}
                  alt=""
                  className="max-h-40 w-auto object-contain shadow-[0_16px_30px_rgba(15,23,42,0.18)]"
                />
              </div>
            )}
            <div className="headline-shell">
              <h1 className="headline-wow text-4xl sm:text-6xl">
                <span>{headline}</span>
              </h1>
            </div>
          </div>
          <p className="mx-auto mt-3 max-w-2xl text-base text-slate-600 sm:text-lg">
            {message}
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <button
              type="button"
              onClick={() => setIsFullscreen(true)}
              className="rounded-full bg-[#111827] px-5 py-2 text-[11px] uppercase tracking-[0.35em] text-white shadow hover:bg-slate-900"
            >
              Fullscreen
            </button>
            <button
              type="button"
              onClick={() => setIsConfetti((prev) => !prev)}
              className="rounded-full border border-slate-200 bg-white px-5 py-2 text-[11px] uppercase tracking-[0.35em] text-slate-700"
            >
              {isConfetti ? "Hide confetti" : "Show confetti"}
            </button>
          </div>
        </header>
        <div className="pointer-events-none fixed inset-x-0 top-6 z-50 flex justify-center">
          <button
            type="button"
            onClick={() => setIsMuted((prev) => !prev)}
            className="group pointer-events-auto relative flex items-center rounded-full border border-slate-200 bg-white/90 p-3 text-slate-700 shadow-[0_10px_24px_rgba(15,23,42,0.12)] backdrop-blur transition-all duration-300 ease-in-out hover:bg-white hover:pr-5"
            aria-label={isMuted ? "Unmute" : "Mute"}
          >
            {isMuted ? (
              <VolumeX size={20} className="text-slate-600" />
            ) : (
              <Volume2 size={20} className="text-slate-600" />
            )}
            <span className="max-w-0 overflow-hidden whitespace-nowrap text-[11px] text-slate-700 opacity-0 transition-all duration-300 ease-in-out group-hover:ml-2 group-hover:max-w-xs group-hover:opacity-100">
              M = ปิด/เปิดเสียง
            </span>
          </button>
        </div>

        <section className="grid gap-8 lg:grid-cols-1">
          <div className="rounded-[32px] border border-slate-100 bg-white p-6 shadow-[0_18px_40px_rgba(15,23,42,0.08)]">
            <h2 className="text-xs font-semibold uppercase tracking-[0.35em] text-slate-400">
              Customize
            </h2>
            <div className="mt-4 grid gap-4">
              <label className="text-[11px] uppercase tracking-[0.35em] text-slate-400">
                Name
                <input
                  type="text"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700"
                />
              </label>
              <label className="text-[11px] uppercase tracking-[0.35em] text-slate-400">
                Message
                <textarea
                  rows={3}
                  value={message}
                  onChange={(event) => setMessage(event.target.value)}
                  className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700"
                />
              </label>
              <label className="text-[11px] uppercase tracking-[0.35em] text-slate-400">
                Background (Image/Video)
                <input
                  type="file"
                  accept="image/*,video/*"
                  onChange={handleBgFile}
                  className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 file:mr-3 file:rounded-full file:border-0 file:bg-slate-900 file:px-3 file:py-1 file:text-[11px] file:uppercase file:tracking-[0.3em] file:text-white"
                />
              </label>
              <label className="text-[11px] uppercase tracking-[0.35em] text-slate-400">
                Hero Image
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleHeroImage}
                  className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 file:mr-3 file:rounded-full file:border-0 file:bg-slate-900 file:px-3 file:py-1 file:text-[11px] file:uppercase file:tracking-[0.3em] file:text-white"
                />
              </label>
              {bgUrl && bgType === "video" && (
                <div>
                  <div className="mb-2 text-[11px] uppercase tracking-[0.35em] text-slate-400">
                    Crop Video
                  </div>
                  <div
                    ref={cropBoxRef}
                    className="relative h-40 w-full overflow-hidden rounded-2xl border border-slate-200 bg-black"
                  >
                    <video
                      src={bgUrl}
                      autoPlay
                      loop
                      muted
                      playsInline
                      className="absolute inset-0 h-full w-full object-contain opacity-80"
                    />
                    <div
                      className="crop-frame"
                      style={{
                        left: `${crop.x}%`,
                        top: `${crop.y}%`,
                        width: `${crop.w}%`,
                        height: `${crop.h}%`,
                      }}
                      onPointerDown={startCropDrag}
                    >
                      <span className="crop-handle nw" onPointerDown={startCropResize("nw")} />
                      <span className="crop-handle ne" onPointerDown={startCropResize("ne")} />
                      <span className="crop-handle sw" onPointerDown={startCropResize("sw")} />
                      <span className="crop-handle se" onPointerDown={startCropResize("se")} />
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setCrop({ x: 10, y: 10, w: 80, h: 80 })}
                    className="mt-3 rounded-full border border-slate-200 bg-white px-4 py-2 text-[11px] uppercase tracking-[0.3em] text-slate-700"
                  >
                    Reset Crop
                  </button>
                </div>
              )}
            </div>
          </div>
        </section>
      </div>


      {isFullscreen && (
        <div className="fixed inset-0 z-50 bg-white">
          {bgUrl && bgType === "image" && (
            <img
              src={bgUrl}
              alt=""
              className="pointer-events-none absolute inset-0 -z-10 h-full w-full object-cover opacity-60"
            />
          )}
          {bgUrl && bgType === "video" && (
            <div className="bg-tile pointer-events-none -z-10">
              {Array.from({ length: 9 }).map((_, i) => (
                <video
                  key={`tile-full-${i}`}
                  ref={i === 0 ? videoRef : null}
                  src={bgUrl}
                  autoPlay
                  loop
                  muted={i === 0 ? isMuted : true}
                  playsInline
                  style={{
                    transformOrigin: "top left",
                    transform: `translate(${-crop.x}%, ${-crop.y}%) scale(${100 / crop.w}, ${100 / crop.h})`,
                  }}
                />
              ))}
            </div>
          )}
          <div className={`pointer-events-none absolute inset-0 -z-0 ${overlayClass}`} />
          <div className="absolute right-6 top-6 flex gap-2">
            {/* <button
              type="button"
              onClick={() => setIsFullscreen(false)}
              className="rounded-full bg-slate-900 px-4 py-2 text-[10px] uppercase tracking-[0.3em] text-white"
            >
              Exit Fullscreen (Esc)
            </button> */}
          </div>
          <div className="flex h-full items-center justify-center px-10 text-center">
            <div className="floaty">
              {heroImage && (
                <div className="mb-5 flex justify-center">
                  <img
                    src={heroImage}
                    alt=""
                    className="max-h-56 w-auto object-contain shadow-[0_16px_30px_rgba(15,23,42,0.18)]"
                  />
                </div>
              )}
              <h1 className="headline-wow text-4xl sm:text-7xl">
                <span>{headline}</span>
              </h1>
              <p className="mt-5 text-lg text-slate-700 sm:text-2xl max-w-2xl mx-auto">{message}</p>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
