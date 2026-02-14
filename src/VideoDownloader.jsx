import { useEffect, useMemo, useRef, useState } from "react";

const rawApiBase = (import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");
const browserHost = typeof window !== "undefined" ? window.location.hostname : "";
const isLocalBrowser = browserHost === "localhost" || browserHost === "127.0.0.1";
const pointsToLocalBackend = /localhost|127\.0\.0\.1/.test(rawApiBase);
const API_BASE = !isLocalBrowser && pointsToLocalBackend ? "" : rawApiBase;
const VIDEO_API_PREFIX = API_BASE ? `${API_BASE}/video` : "/api/video";

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "-";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

function guessFilename(title, fallbackExt) {
  if (!title) return `video.${fallbackExt}`;
  const sanitized = title
    .replace(/[\\/:*?"<>|]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return sanitized ? `${sanitized}.${fallbackExt}` : `video.${fallbackExt}`;
}

function buildFallbackWaveformBars(count = 110) {
  return Array.from({ length: count }, (_, i) => {
    const a = Math.abs(Math.sin(i * 0.22));
    const b = Math.abs(Math.cos(i * 0.075));
    return Math.round((a * 0.55 + b * 0.45) * 100);
  });
}

export default function VideoDownloader() {
  const [url, setUrl] = useState("");
  const [format, setFormat] = useState("mp4");
  const [filename, setFilename] = useState("");
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);
  const [volume, setVolume] = useState(1);
  const [speed, setSpeed] = useState(1);
  const [fadeIn, setFadeIn] = useState(0);
  const [fadeOut, setFadeOut] = useState(0);
  const [livePreviewEnabled, setLivePreviewEnabled] = useState(true);

  const [info, setInfo] = useState(null);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState("");
  const [isChecking, setIsChecking] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewKind, setPreviewKind] = useState("audio");
  const [waveformBars, setWaveformBars] = useState(() => buildFallbackWaveformBars(110));

  const autoCheckRef = useRef({ timer: null, lastUrl: "" });
  const livePreviewTimerRef = useRef(null);
  const previewSeqRef = useRef(0);
  const previewFnRef = useRef(null);
  const waveformRef = useRef(null);
  const dragStateRef = useRef({ active: false });
  const mediaPreviewRef = useRef(null);
  const previewResumeTimeRef = useRef(0);
  const previewAutoPlayRef = useRef(false);
  const [isWaveDragging, setIsWaveDragging] = useState(false);

  const suggestedName = useMemo(() => {
    if (filename.trim()) return filename.trim();
    return guessFilename(info?.title, format);
  }, [filename, info, format]);

  const isAudioFormat = format === "mp3" || format === "wav";
  const timelineMax = Math.max(10, Number(info?.duration || 300));
  const endMarkerValue = trimEnd > 0 ? trimEnd : timelineMax;
  const hasCustomEnd = endMarkerValue < timelineMax - 0.05;
  const trimStartPct = Math.min(100, Math.max(0, (trimStart / timelineMax) * 100));
  const trimEndPct = Math.min(100, Math.max(0, (endMarkerValue / timelineMax) * 100));
  const selectionLeftPct = Math.min(trimStartPct, trimEndPct);
  const selectionWidthPct = Math.max(0, trimEndPct - trimStartPct);

  const buildEditPayload = (target) => {
    const startTime = trimStart > 0 ? trimStart : null;
    const endTime = hasCustomEnd ? endMarkerValue : null;
    return {
      url: target,
      format,
      start_time: startTime,
      end_time: endTime,
      volume,
      speed,
      fade_in: fadeIn,
      fade_out: fadeOut,
    };
  };

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  const handleWavePointerDown = (event) => {
    if (!waveformRef.current) return;
    const rect = waveformRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;

    const sec = clamp(((event.clientX - rect.left) / rect.width) * timelineMax, 0, timelineMax);
    const tolerance = Math.max(0.15, timelineMax * 0.015);
    const start = trimStart;
    const end = endMarkerValue;

    let mode = "end";
    if (Math.abs(sec - start) <= tolerance) {
      mode = "start";
    } else if (Math.abs(sec - end) <= tolerance) {
      mode = "end";
    } else if (sec > start && sec < end) {
      mode = "range";
    } else {
      mode = sec < start ? "start" : "end";
    }

    dragStateRef.current = {
      active: true,
      pointerId: event.pointerId,
      mode,
      rectLeft: rect.left,
      rectWidth: rect.width,
      startStart: start,
      startEnd: end,
      startSec: sec,
    };
    setIsWaveDragging(true);
    event.preventDefault();
  };

  const requestInfo = async (target, signal) => {
    const res = await fetch(`${VIDEO_API_PREFIX}/info`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: target }),
      signal,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    return res.json();
  };

  const updateWaveformBarsFromBlob = async (blob, requestId) => {
    try {
      const arrayBuffer = await blob.arrayBuffer();
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const audioContext = new AudioCtx();
      try {
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        const source = audioBuffer.getChannelData(0);
        const bars = 110;
        const block = Math.max(1, Math.floor(source.length / bars));
        const nextBars = Array.from({ length: bars }, (_, i) => {
          const start = i * block;
          const end = Math.min(source.length, start + block);
          let peak = 0;
          for (let j = start; j < end; j += 1) {
            const v = Math.abs(source[j]);
            if (v > peak) peak = v;
          }
          return Math.round(Math.max(8, peak * 100));
        });
        if (requestId === previewSeqRef.current) {
          setWaveformBars(nextBars);
        }
      } finally {
        await audioContext.close();
      }
    } catch {
      if (requestId === previewSeqRef.current) {
        setWaveformBars(buildFallbackWaveformBars(110));
      }
    }
  };

  const handleCheck = async () => {
    const target = url.trim();
    if (!target) {
      setError("Please enter a video URL.");
      return;
    }
    setError("");
    setIsChecking(true);
    setStatus("Checking URL...");
    try {
      const data = await requestInfo(target);
      setInfo(data);
      setStatus("Ready to download");
    } catch (err) {
      setInfo(null);
      setStatus("Check failed");
      setError(err.message || "Cannot check this URL.");
    } finally {
      setIsChecking(false);
    }
  };

  const handlePreview = async ({ silent = false } = {}) => {
    const target = url.trim();
    if (!target) {
      if (!silent) setError("Please enter a video URL.");
      return;
    }
    const payload = buildEditPayload(target);
    const startTime = payload.start_time;
    const endTime = payload.end_time;
    if (endTime !== null && endTime <= startTime) {
      if (!silent) setError("End time must be greater than start time.");
      return;
    }
    if (!silent) {
      setError("");
      setStatus("Generating preview...");
    }

    const requestId = ++previewSeqRef.current;
    setIsPreviewing(true);
    try {
      const res = await fetch(`${VIDEO_API_PREFIX}/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      if (requestId !== previewSeqRef.current) return;

      if (mediaPreviewRef.current) {
        previewResumeTimeRef.current = mediaPreviewRef.current.currentTime || 0;
      }
      previewAutoPlayRef.current = livePreviewEnabled || !silent;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      const objectUrl = URL.createObjectURL(blob);
      setPreviewUrl(objectUrl);
      setPreviewKind(format === "mp4" ? "video" : "audio");
      void updateWaveformBarsFromBlob(blob, requestId);
      if (!silent) setStatus("Preview ready");
    } catch (err) {
      if (!silent) {
        setStatus("Preview failed");
        setError(err.message || "Preview failed");
      }
    } finally {
      if (requestId === previewSeqRef.current) {
        setIsPreviewing(false);
      }
    }
  };

  const handleDownload = async () => {
    const target = url.trim();
    if (!target) {
      setError("Please enter a video URL.");
      return;
    }
    const payload = buildEditPayload(target);
    const startTime = payload.start_time;
    const endTime = payload.end_time;
    if (endTime !== null && endTime <= startTime) {
      setError("End time must be greater than start time.");
      return;
    }
    setError("");
    setIsDownloading(true);
    setStatus("Downloading...");
    try {
      const res = await fetch(`${VIDEO_API_PREFIX}/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = suggestedName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(objectUrl);
      setStatus("Download complete");
    } catch (err) {
      setStatus("Download failed");
      setError(err.message || "Download failed");
    } finally {
      setIsDownloading(false);
    }
  };

  useEffect(() => {
    previewFnRef.current = handlePreview;
  });

  const handleClear = () => {
    setUrl("");
    setFormat("mp4");
    setFilename("");
    setTrimStart(0);
    setTrimEnd(0);
    setVolume(1);
    setSpeed(1);
    setFadeIn(0);
    setFadeOut(0);
    setWaveformBars(buildFallbackWaveformBars(110));
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl("");
    setPreviewKind("audio");
    setInfo(null);
    setStatus("Ready");
    setError("");
  };

  useEffect(() => {
    if (!info?.duration) return;
    setTrimStart(0);
    setTrimEnd(Math.floor(info.duration * 10) / 10);
  }, [info?.duration]);

  useEffect(() => {
    const onPointerMove = (event) => {
      const drag = dragStateRef.current;
      if (!drag.active) return;
      if (drag.pointerId !== undefined && event.pointerId !== drag.pointerId) return;

      const sec = clamp(((event.clientX - drag.rectLeft) / drag.rectWidth) * timelineMax, 0, timelineMax);
      if (drag.mode === "start") {
        const nextStart = clamp(sec, 0, drag.startEnd - 0.1);
        setTrimStart(nextStart);
        return;
      }
      if (drag.mode === "end") {
        const nextEnd = clamp(sec, drag.startStart + 0.1, timelineMax);
        setTrimEnd(nextEnd);
        return;
      }

      const len = Math.max(0.1, drag.startEnd - drag.startStart);
      const delta = sec - drag.startSec;
      const nextStart = clamp(drag.startStart + delta, 0, timelineMax - len);
      const nextEnd = nextStart + len;
      setTrimStart(nextStart);
      setTrimEnd(nextEnd);
    };

    const onPointerUp = (event) => {
      const drag = dragStateRef.current;
      if (!drag.active) return;
      if (drag.pointerId !== undefined && event.pointerId !== drag.pointerId) return;
      dragStateRef.current = { active: false };
      setIsWaveDragging(false);
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
    };
  }, [timelineMax]);

  useEffect(() => {
    const target = url.trim();
    if (!target || isDownloading) {
      setInfo(null);
      return undefined;
    }

    if (autoCheckRef.current.timer) {
      clearTimeout(autoCheckRef.current.timer);
    }

    const timerId = setTimeout(async () => {
      if (autoCheckRef.current.lastUrl === target) return;
      autoCheckRef.current.lastUrl = target;
      setError("");
      setIsChecking(true);
      setStatus("Checking URL...");
      const controller = new AbortController();
      try {
        const data = await requestInfo(target, controller.signal);
        setInfo(data);
        setStatus("Ready to download");
      } catch (err) {
        if (err.name !== "AbortError") {
          setInfo(null);
          setStatus("Check failed");
          setError(err.message || "Cannot check this URL.");
        }
      } finally {
        setIsChecking(false);
      }
    }, 700);
    autoCheckRef.current.timer = timerId;

    return () => clearTimeout(timerId);
  }, [url, isDownloading]);

  useEffect(() => {
    if (!livePreviewEnabled) return undefined;
    if (!url.trim() || !info) return undefined;
    if (isDownloading || isChecking) return undefined;

    if (livePreviewTimerRef.current) {
      clearTimeout(livePreviewTimerRef.current);
    }

    livePreviewTimerRef.current = setTimeout(() => {
      if (previewFnRef.current) {
        previewFnRef.current({ silent: true });
      }
    }, 450);

    return () => {
      if (livePreviewTimerRef.current) {
        clearTimeout(livePreviewTimerRef.current);
      }
    };
  }, [
    livePreviewEnabled,
    url,
    info,
    format,
    trimStart,
    trimEnd,
    volume,
    speed,
    fadeIn,
    fadeOut,
    isDownloading,
    isChecking,
  ]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (livePreviewTimerRef.current) clearTimeout(livePreviewTimerRef.current);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!previewUrl || !mediaPreviewRef.current) return;
    const media = mediaPreviewRef.current;
    const seekTo = previewResumeTimeRef.current;
    const tryPlay = async () => {
      if (seekTo > 0) {
        media.currentTime = seekTo;
      }
      if (previewAutoPlayRef.current) {
        try {
          await media.play();
        } catch {
          // Browser may block autoplay until user interaction.
        }
      }
    };
    void tryPlay();
  }, [previewUrl]);

  return (
    <main className="relative min-h-screen overflow-hidden bg-[#0b1225] font-[Space_Grotesk] text-slate-100">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-24 left-1/2 h-72 w-72 -translate-x-1/2 rounded-full bg-cyan-400/25 blur-[120px]" />
        <div className="absolute bottom-0 right-0 h-80 w-80 rounded-full bg-emerald-300/20 blur-[140px]" />
      </div>

      <div className="relative mx-auto max-w-5xl px-6 pb-12 pt-24 sm:px-10">
        <header className="mb-8 rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl">
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">Video Tools</p>
          <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">Video / Sound Downloader</h1>
          <p className="mt-3 text-sm text-slate-300">Download and edit MP4, MP3, or WAV with timeline controls.</p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
            <label className="text-[11px] uppercase tracking-[0.35em] text-slate-400">
              Video URL
              <div className="mt-2 flex items-center gap-2">
                <input
                  type="url"
                  value={url}
                  onChange={(event) => setUrl(event.target.value)}
                  placeholder="Paste video URL..."
                  className="w-full rounded-2xl border border-white/10 bg-[#0f172a] px-3 py-3 text-sm text-slate-100 placeholder:text-slate-500"
                />
                <button
                  type="button"
                  onClick={handleClear}
                  className="flex h-11 w-14 items-center justify-center rounded-2xl border border-white/20 bg-white/5 text-slate-200 transition hover:border-rose-300/60 hover:text-rose-100"
                  aria-label="Clear"
                >
                  <img src="/clear.png" alt="" className="h-8 w-8" />
                </button>
              </div>
            </label>

            <label className="mt-5 block text-[11px] uppercase tracking-[0.35em] text-slate-400">
              File Format
              <div className="mt-2 flex gap-3">
                {["mp4", "mp3", "wav"].map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => setFormat(item)}
                    className={`rounded-full px-4 py-2 text-[13px] uppercase tracking-[0.1em] transition ${
                      format === item
                        ? "bg-cyan-300 text-slate-900"
                        : "border border-white/20 bg-white/5 text-slate-200 hover:border-cyan-300/60"
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </label>

            <label className="mt-5 block text-[11px] uppercase tracking-[0.35em] text-slate-400">
              File Name
              <input
                type="text"
                value={filename}
                onChange={(event) => setFilename(event.target.value)}
                placeholder={suggestedName}
                className="mt-2 w-full rounded-2xl border border-white/10 bg-[#0f172a] px-3 py-3 text-sm text-slate-100 placeholder:text-slate-500"
              />
            </label>

            <div className="mt-5 rounded-2xl border border-white/10 bg-[#0f172a] p-4">
              <div className="flex items-center justify-between">
                <p className="text-[11px] uppercase tracking-[0.35em] text-cyan-300/85">Studio Edit Deck</p>
                <span className="rounded-full border border-cyan-300/40 bg-cyan-400/10 px-3 py-1 text-[10px] text-cyan-100">
                  {format.toUpperCase()}
                </span>
              </div>

              <div className="mt-2 flex items-center justify-between rounded-xl border border-cyan-300/35 bg-cyan-400/10 px-3 py-2 text-[11px]">
                <span>Auto Preview</span>
                <button
                  type="button"
                  onClick={() => setLivePreviewEnabled((v) => !v)}
                  className={`rounded-full px-3 py-1 text-[10px] uppercase tracking-[0.12em] ${
                    livePreviewEnabled
                      ? "bg-emerald-400/25 text-emerald-100"
                      : "border border-white/20 bg-white/5 text-slate-300"
                  }`}
                >
                  {livePreviewEnabled ? "On" : "Off"}
                </button>
              </div>

              <div className="mt-3 rounded-xl border border-white/10 bg-[#111827] p-3">
                <div
                  ref={waveformRef}
                  onPointerDown={handleWavePointerDown}
                  className={`relative h-20 overflow-hidden rounded-lg bg-gradient-to-b from-[#0d152d] to-[#091021] ${
                    isWaveDragging ? "cursor-grabbing" : "cursor-ew-resize"
                  }`}
                >
                  <div className="absolute inset-0 flex items-end gap-[2px] px-1 pb-1">
                    {waveformBars.map((height, index) => (
                      <span
                        key={`w-${index}`}
                        className="block w-[3px] rounded-t bg-cyan-300/70"
                        style={{ height: `${height}%` }}
                      />
                    ))}
                  </div>
                  <div
                    className="pointer-events-none absolute inset-y-0 bg-cyan-300/12"
                    style={{ left: `${selectionLeftPct}%`, width: `${selectionWidthPct}%` }}
                  />
                  <div className="pointer-events-none absolute inset-y-0 w-[2px] bg-emerald-300" style={{ left: `${trimStartPct}%` }} />
                  <div className="pointer-events-none absolute inset-y-0 w-[2px] bg-rose-300" style={{ left: `${trimEndPct}%` }} />
                </div>

                <div className="mt-3 grid gap-3">
                  <div className="flex items-center justify-between rounded-lg border border-white/10 bg-[#0b1329] px-3 py-2 text-[11px] text-slate-300">
                    <span>Selected Range</span>
                    <span className="font-semibold text-cyan-100">
                      {trimStart.toFixed(1)}s - {endMarkerValue.toFixed(1)}s
                    </span>
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-slate-400">
                      <span>Start</span>
                      <span>{trimStart.toFixed(1)}s</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max={timelineMax}
                      step="0.1"
                      value={trimStart}
                      onChange={(event) => {
                        const nextStart = Number(event.target.value);
                        setTrimStart(nextStart);
                        if (endMarkerValue <= nextStart) {
                          setTrimEnd(Math.min(timelineMax, nextStart + 0.1));
                        }
                      }}
                      className="w-full accent-emerald-300"
                    />
                  </div>
                  <div>
                    <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-slate-400">
                      <span>End</span>
                      <span>{endMarkerValue.toFixed(1)}s</span>
                    </div>
                    <input
                      type="range"
                      min={Math.min(timelineMax, trimStart + 0.1)}
                      max={timelineMax}
                      step="0.1"
                      value={endMarkerValue}
                      onChange={(event) => setTrimEnd(Math.max(trimStart + 0.1, Number(event.target.value)))}
                      className="w-full accent-rose-300"
                    />
                  </div>
                </div>
              </div>

              <div className={`mt-3 grid gap-3 sm:grid-cols-2 ${!isAudioFormat ? "opacity-60" : ""}`}>
                <label className="text-[10px] uppercase tracking-[0.22em] text-slate-400">
                  Volume: {volume.toFixed(2)}
                  <input
                    type="range"
                    min="0.1"
                    max="3"
                    step="0.1"
                    value={volume}
                    onChange={(event) => setVolume(Number(event.target.value))}
                    disabled={!isAudioFormat}
                    className="mt-2 w-full accent-cyan-300 disabled:opacity-40"
                  />
                </label>
                <label className="text-[10px] uppercase tracking-[0.22em] text-slate-400">
                  Speed: {speed.toFixed(2)}x
                  <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.05"
                    value={speed}
                    onChange={(event) => setSpeed(Number(event.target.value))}
                    disabled={!isAudioFormat}
                    className="mt-2 w-full accent-cyan-300 disabled:opacity-40"
                  />
                </label>
                <label className="text-[10px] uppercase tracking-[0.22em] text-slate-400">
                  Fade In: {fadeIn.toFixed(1)}s
                  <input
                    type="range"
                    min="0"
                    max="30"
                    step="0.1"
                    value={fadeIn}
                    onChange={(event) => setFadeIn(Number(event.target.value))}
                    disabled={!isAudioFormat}
                    className="mt-2 w-full accent-cyan-300 disabled:opacity-40"
                  />
                </label>
                <label className="text-[10px] uppercase tracking-[0.22em] text-slate-400">
                  Fade Out: {fadeOut.toFixed(1)}s
                  <input
                    type="range"
                    min="0"
                    max="30"
                    step="0.1"
                    value={fadeOut}
                    onChange={(event) => setFadeOut(Number(event.target.value))}
                    disabled={!isAudioFormat}
                    className="mt-2 w-full accent-cyan-300 disabled:opacity-40"
                  />
                </label>
              </div>
            </div>

            <div className="mt-6 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={handleCheck}
                disabled={isChecking}
                className="rounded-full border border-white/20 bg-white/5 px-5 py-2 text-[11px] uppercase tracking-[0.35em] text-slate-100 transition hover:border-cyan-300/60 disabled:opacity-60"
              >
                {isChecking ? "Checking..." : "Check"}
              </button>
              <button
                type="button"
                onClick={() => handlePreview({ silent: false })}
                disabled={isPreviewing}
                className="rounded-full border border-cyan-300/50 bg-cyan-400/20 px-5 py-2 text-[11px] uppercase tracking-[0.35em] text-cyan-100 transition hover:bg-cyan-300/25 disabled:opacity-60"
              >
                {isPreviewing ? "Previewing..." : "Preview"}
              </button>
              <button
                type="button"
                onClick={handleDownload}
                disabled={isDownloading}
                className="rounded-full bg-cyan-400/90 px-5 py-2 text-[11px] uppercase tracking-[0.35em] text-slate-900 transition hover:bg-cyan-300 disabled:opacity-60"
              >
                {isDownloading ? "Downloading..." : "Download"}
              </button>
            </div>

            {previewUrl && (
              <div className="mt-4 overflow-hidden rounded-2xl border border-cyan-300/30 bg-[#0b1329] p-3">
                <p className="mb-2 text-[10px] uppercase tracking-[0.28em] text-cyan-200/85">Preview Player</p>
                {previewKind === "video" ? (
                  <video ref={mediaPreviewRef} src={previewUrl} controls className="h-56 w-full rounded-xl bg-black" />
                ) : (
                  <audio ref={mediaPreviewRef} src={previewUrl} controls className="w-full" />
                )}
              </div>
            )}

            {error && <p className="mt-4 text-sm text-rose-300">{error}</p>}
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-xl">
            <p className="text-xs uppercase tracking-[0.35em] text-slate-400">Result</p>
            <div className="mt-4 rounded-2xl border border-white/10 bg-[#0f172a] px-4 py-3 text-sm text-slate-200">
              {status}
            </div>

            <div className="mt-6 space-y-3 text-sm text-slate-300">
              <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                <span>Title</span>
                <span className="text-slate-100">{info?.title || "-"}</span>
              </div>
              <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                <span>Duration</span>
                <span className="text-slate-100">{formatDuration(info?.duration)}</span>
              </div>
              <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                <span>Platform</span>
                <span className="text-slate-100">{info?.extractor || "-"}</span>
              </div>
            </div>

            {info?.thumbnail && (
              <div className="mt-5 overflow-hidden rounded-2xl border border-white/10">
                <img src={info.thumbnail} alt="thumbnail" className="h-44 w-full object-cover" />
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
