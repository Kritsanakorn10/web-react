import { useEffect, useMemo, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

const defaultParams = {
  time_step: 7,
  train_ratio: 0.8,
  max_epochs: 1500,
  batch_size: 32,
  learning_rate: 0.01,
};

export default function MLToolkit() {
  const [params, setParams] = useState(defaultParams);
  const [paramInputs, setParamInputs] = useState({
    time_step: String(defaultParams.time_step),
    train_ratio: String(defaultParams.train_ratio),
    max_epochs: String(defaultParams.max_epochs),
    batch_size: String(defaultParams.batch_size),
    learning_rate: String(defaultParams.learning_rate),
  });
  const [hiddenInputs, setHiddenInputs] = useState(
    Array.from({ length: 10 }, (_, index) => (index === 0 ? "50" : "0"))
  );
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);

  const testMetrics = result?.metrics?.test;
  const trainMetrics = result?.metrics?.train;
  const testSeries = result?.series?.test;
  const trainSeries = result?.series?.train;

  const rmse = useMemo(() => {
    if (!testMetrics) return null;
    return testMetrics.rmse.toFixed(3);
  }, [testMetrics]);

  const clamp = (field, value) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return defaultParams[field] ?? num;
    if (field === "time_step") return Math.min(Math.max(num, 2), 30);
    if (field === "train_ratio") return Math.min(Math.max(num, 0.5), 0.95);
    if (field === "max_epochs") return Math.min(Math.max(num, 10), 2000);
    if (field === "batch_size") return Math.min(Math.max(num, 4), 256);
    if (field === "learning_rate") return Math.min(Math.max(num, 0.00001), 0.5);
    return num;
  };

  const handleChange = (field) => (event) => {
    const value = event.target.value;
    setParamInputs((prev) => ({ ...prev, [field]: value }));
  };

  const handleBlur = (field) => () => {
    const value = paramInputs[field];
    const clamped = clamp(field, value);
    setParams((prev) => ({ ...prev, [field]: clamped }));
    setParamInputs((prev) => ({ ...prev, [field]: String(clamped) }));
  };

  const handleHiddenChange = (index) => (event) => {
    const value = event.target.value;
    setHiddenInputs((prev) => prev.map((item, i) => (i === index ? value : item)));
  };

  const handleHiddenBlur = (index) => () => {
    const raw = hiddenInputs[index];
    const num = Number(raw);
    const clamped = Number.isFinite(num) ? Math.min(Math.max(num, 0), 256) : 0;
    setHiddenInputs((prev) =>
      prev.map((item, i) => (i === index ? String(clamped) : item))
    );
  };

  const isTraining = status === "Training...";

  const handleTrain = async () => {
    setError("");
    setStatus("Training...");
    try {
      if (!file) {
        setError("กรุณาอัปโหลดไฟล์ก่อน");
        setStatus("Failed");
        return;
      }
      const formData = new FormData();
      formData.append("file", file);
      Object.entries(params).forEach(([key, value]) => {
        formData.append(key, String(value));
      });
      hiddenInputs.forEach((value) => {
        formData.append("hidden_units", String(value || 0));
      });

      const res = await fetch("http://127.0.0.1:8000/train", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || "Train failed");
      }
      const data = await res.json();
      setJobId(data.job_id);
      setProgress(0);
      setCurrentEpoch(0);
      setTotalEpochs(params.max_epochs);
    } catch (err) {
      setError("เทรนไม่สำเร็จ: โปรดตรวจสอบว่า backend เปิดอยู่");
      setStatus("Failed");
    }
  };

  const handleStop = async () => {
    if (!jobId) return;
    try {
      await fetch(`http://127.0.0.1:8000/train/${jobId}/cancel`, { method: "POST" });
      setStatus("Cancelled");
    } catch {
      setError("หยุดไม่สำเร็จ");
    }
  };

  useEffect(() => {
    if (!jobId) return;
    let timer = null;

    const poll = async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/train/${jobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "done") {
          setResult(data.result);
          setStatus("Done");
          setJobId(null);
          setProgress(1);
          setCurrentEpoch(data.total_epochs ?? totalEpochs);
          return;
        }
        if (data.status === "failed") {
          setError(data.error || "เทรนไม่สำเร็จ");
          setStatus("Failed");
          setJobId(null);
          setProgress(0);
          return;
        }
        if (data.status === "cancelled") {
          setStatus("Cancelled");
          setJobId(null);
          setProgress(0);
          setCurrentEpoch(0);
          return;
        }
        setStatus("Training...");
        setProgress(data.progress ?? 0);
        setCurrentEpoch(data.current_epoch ?? 0);
        setTotalEpochs(data.total_epochs ?? totalEpochs);
        timer = setTimeout(poll, 1000);
      } catch {
        setError("ไม่สามารถเชื่อมต่อ backend");
        setStatus("Failed");
        setJobId(null);
        setProgress(0);
        setCurrentEpoch(0);
      }
    };

    poll();
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  return (
    <main className="relative min-h-screen bg-[#0a0f1f] text-slate-100 font-[Space_Grotesk] overflow-hidden">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-24 left-1/2 h-72 w-72 -translate-x-1/2 rounded-full bg-cyan-400/25 blur-[120px] motion-safe:animate-pulse" />
        <div className="absolute bottom-0 right-0 h-80 w-80 rounded-full bg-lime-300/20 blur-[140px]" />
        <div className="absolute top-1/3 -left-10 h-64 w-64 rounded-full bg-sky-500/20 blur-[110px]" />
      </div>

      <div className="relative mx-auto max-w-5xl px-6 pb-12 pt-24 sm:px-10">
        <header className="mb-10 rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl">
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">
            ML Training Lab
          </p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white sm:text-5xl">
            ระบบเทรน/เทสโมเดล LSTM
          </h1>
          <p className="mt-2 text-sm text-slate-300">
            เทรนโมเดลจากไฟล์ข้อมูลบนเซิร์ฟเวอร์ และปรับพารามิเตอร์ได้แบบเรียลไทม์ ทำไปทำหอกอะไรแม่เย็ด
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-7 backdrop-blur-xl">
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
              Training Parameters
            </h2>
            <label className="mt-4 block text-xs uppercase tracking-[0.25em] text-slate-400">
              Upload data (.xlsx)
              <input
                type="file"
                accept=".xlsx,.xls"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                className="mt-2 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200 file:mr-4 file:rounded-full file:border-0 file:bg-cyan-300/20 file:px-3 file:py-1 file:text-xs file:uppercase file:tracking-[0.3em] file:text-cyan-100"
              />
            </label>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <label className="text-xs uppercase tracking-[0.25em] text-slate-400">
                timeStep
                <input
                  type="number"
                  min="2"
                  max="30"
                  value={paramInputs.time_step}
                  onChange={handleChange("time_step")}
                  onBlur={handleBlur("time_step")}
                  className="mt-2 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200"
                />
              </label>
              <label className="text-xs uppercase tracking-[0.25em] text-slate-400">
                train_ratio
                <input
                  type="number"
                  step="0.01"
                  min="0.5"
                  max="0.95"
                  value={paramInputs.train_ratio}
                  onChange={handleChange("train_ratio")}
                  onBlur={handleBlur("train_ratio")}
                  className="mt-2 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200"
                />
              </label>
              <label className="text-xs uppercase tracking-[0.25em] text-slate-400">
                MaxEpochs
                <input
                  type="number"
                  min="10"
                  max="2000"
                  value={paramInputs.max_epochs}
                  onChange={handleChange("max_epochs")}
                  onBlur={handleBlur("max_epochs")}
                  className="mt-2 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200"
                />
              </label>
              <label className="text-xs uppercase tracking-[0.25em] text-slate-400">
                MiniBatchSize
                <input
                  type="number"
                  min="4"
                  max="256"
                  value={paramInputs.batch_size}
                  onChange={handleChange("batch_size")}
                  onBlur={handleBlur("batch_size")}
                  className="mt-2 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200"
                />
              </label>
              <label className="text-xs uppercase tracking-[0.25em] text-slate-400">
                InitialLearnRate
                <input
                  type="number"
                  step="0.001"
                  min="0.00001"
                  max="0.5"
                  value={paramInputs.learning_rate}
                  onChange={handleChange("learning_rate")}
                  onBlur={handleBlur("learning_rate")}
                  className="mt-2 w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-200"
                />
              </label>
              <div className="sm:col-span-2">
                <p className="text-xs uppercase tracking-[0.25em] text-slate-400">
                  Hidden Units Per Layer (0 = skip)
                </p>
                <div className="mt-3 grid gap-3 sm:grid-cols-5">
                  {hiddenInputs.map((value, index) => (
                    <label
                      key={`layer-${index}`}
                      className="text-[10px] uppercase tracking-[0.25em] text-slate-400"
                    >
                      Layer {index + 1}
                      <input
                        type="number"
                        min="0"
                        max="256"
                        value={value}
                        onChange={handleHiddenChange(index)}
                        onBlur={handleHiddenBlur(index)}
                        className="mt-2 w-full rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200"
                      />
                    </label>
                  ))}
                </div>
              </div>
            </div>
            <button
              type="button"
              onClick={handleTrain}
              disabled={isTraining}
              className={`mt-6 flex items-center justify-center gap-2 rounded-2xl border border-white/10 px-5 py-3 text-xs uppercase tracking-[0.35em] text-cyan-100 transition hover:-translate-y-1 hover:border-cyan-300/60 ${
                isTraining ? "bg-cyan-300/10 opacity-70" : "bg-cyan-300/20"
              }`}
            >
              {isTraining && (
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-cyan-200 border-t-transparent" />
              )}
              {isTraining ? "Training..." : "Train & Test"}
            </button>
            <button
              type="button"
              onClick={handleStop}
              disabled={!isTraining}
              className="mt-3 w-full rounded-2xl border border-rose-400/40 bg-rose-400/10 px-5 py-2 text-xs uppercase tracking-[0.35em] text-rose-200 transition hover:border-rose-400/70 disabled:opacity-40"
            >
              Stop
            </button>
            {error && <p className="mt-3 text-xs text-rose-300">{error}</p>}
            {isTraining && (
              <p className="mt-3 text-xs text-slate-400">
                กำลังเทรนและทดสอบโมเดล กรุณารอสักครู่...
              </p>
            )}
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/5 p-7 backdrop-blur-xl">
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-lime-200/90">
              Result
            </h2>
            <div className="mt-4 space-y-3 text-sm text-slate-200">
              <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/10 px-4 py-3">
                <span>Status</span>
                <span className="text-xs text-cyan-200/80">{status}</span>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/10 px-4 py-3">
                <div className="flex items-center justify-between text-xs text-slate-300">
                  <span>Epoch Progress</span>
                  <span>{Math.round(progress * 100)}%</span>
                </div>
                <div className="mt-3 h-2 w-full rounded-full bg-slate-800/70">
                  <div
                    className="h-2 rounded-full bg-gradient-to-r from-cyan-400/80 to-lime-300/80 transition-all duration-300"
                    style={{ width: `${Math.round(progress * 100)}%` }}
                  />
                </div>
                <div className="mt-3 flex items-center justify-between text-xs text-slate-300">
                  <span>
                    Epoch {currentEpoch}/{totalEpochs || params.max_epochs}
                  </span>
                  <span>{isTraining ? "Running" : status}</span>
                </div>
              </div>
              {trainMetrics && (
                <div className="rounded-2xl border border-cyan-300/30 bg-cyan-300/10 px-4 py-3">
                  <div className="flex items-center justify-between text-xs text-cyan-100">
                    <span>RMSE (Train)</span>
                    <span>{trainMetrics.rmse.toFixed(3)}</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-xs text-cyan-100">
                    <span>MAE (Train)</span>
                    <span>{trainMetrics.mae.toFixed(3)}</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-xs text-cyan-100">
                    <span>R2 (Train)</span>
                    <span>{trainMetrics.r2.toFixed(3)}</span>
                  </div>
                </div>
              )}
              {testMetrics && (
                <div className="rounded-2xl border border-lime-300/30 bg-lime-300/10 px-4 py-3">
                  <div className="flex items-center justify-between text-xs text-lime-100">
                    <span>RMSE (Test)</span>
                    <span>{testMetrics.rmse.toFixed(3)}</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-xs text-lime-100">
                    <span>MAE (Test)</span>
                    <span>{testMetrics.mae.toFixed(3)}</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between text-xs text-lime-100">
                    <span>R2 (Test)</span>
                    <span>{testMetrics.r2.toFixed(3)}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>


        {trainSeries && (
          <section className="mt-8 rounded-3xl border border-cyan-300/20 bg-white/5 p-8 backdrop-blur-xl">
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
              Train Prediction Chart
            </h2>
            <div className="mt-6">
              <Line
                data={{
                  labels: trainSeries.actual.map((_, index) => index + 1),
                  datasets: [
                    {
                      label: "Train Actual (mm)",
                      data: trainSeries.actual,
                      borderColor: "rgba(59, 130, 246, 0.95)",
                      backgroundColor: "rgba(59, 130, 246, 0.2)",
                      tension: 0.35,
                    },
                    {
                      label: "Train Predicted (mm)",
                      data: trainSeries.pred,
                      borderColor: "rgba(217, 70, 239, 0.95)",
                      backgroundColor: "rgba(217, 70, 239, 0.2)",
                      tension: 0.35,
                    },
                  ],
                }}
                options={{
                  plugins: { legend: { labels: { color: "#cbd5f5" } } },
                  scales: {
                    x: {
                      title: { display: true, text: "Index", color: "#cbd5f5" },
                      ticks: { color: "#94a3b8" },
                      grid: { color: "rgba(148,163,184,0.1)" },
                    },
                    y: {
                      title: { display: true, text: "Rainfall (mm)", color: "#cbd5f5" },
                      ticks: { color: "#94a3b8" },
                      grid: { color: "rgba(148,163,184,0.1)" },
                    },
                  },
                }}
              />
            </div>
            <div className="mt-8 overflow-hidden rounded-2xl border border-cyan-300/20">
              <div className="grid grid-cols-3 bg-cyan-300/10 px-4 py-2 text-xs uppercase tracking-[0.3em] text-cyan-100">
                <span>Day</span>
                <span>Actual (mm)</span>
                <span>Predicted (mm)</span>
              </div>
              <div className="max-h-64 overflow-auto">
                {trainSeries.actual.map((value, index) => (
                  <div
                    key={`train-row-${index}`}
                    className={`grid grid-cols-3 border-t border-cyan-300/10 px-4 py-2 text-xs text-slate-200 ${
                      index % 2 === 0 ? "bg-cyan-300/5" : "bg-white/0"
                    }`}
                  >
                    <span>{index + 1}</span>
                    <span>{value.toFixed(3)}</span>
                    <span>{trainSeries.pred[index]?.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {testSeries && (
          <section className="mt-8 rounded-3xl border border-lime-300/20 bg-white/5 p-8 backdrop-blur-xl">
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-cyan-300/80">
              Test Prediction Chart
            </h2>
            <div className="mt-6">
              <Line
                data={{
                  labels: testSeries.actual.map((_, index) => index + 1),
                  datasets: [
                    {
                      label: "Test Actual (mm)",
                      data: testSeries.actual,
                      borderColor: "rgba(34, 197, 94, 0.95)",
                      backgroundColor: "rgba(34, 197, 94, 0.2)",
                      tension: 0.35,
                    },
                    {
                      label: "Test Predicted (mm)",
                      data: testSeries.pred,
                      borderColor: "rgba(251, 146, 60, 0.95)",
                      backgroundColor: "rgba(251, 146, 60, 0.2)",
                      tension: 0.35,
                    },
                  ],
                }}
                options={{
                  plugins: { legend: { labels: { color: "#cbd5f5" } } },
                  scales: {
                    x: {
                      title: { display: true, text: "Index", color: "#cbd5f5" },
                      ticks: { color: "#94a3b8" },
                      grid: { color: "rgba(148,163,184,0.1)" },
                    },
                    y: {
                      title: { display: true, text: "Rainfall (mm)", color: "#cbd5f5" },
                      ticks: { color: "#94a3b8" },
                      grid: { color: "rgba(148,163,184,0.1)" },
                    },
                  },
                }}
              />
            </div>
            <div className="mt-8 overflow-hidden rounded-2xl border border-lime-300/20">
              <div className="grid grid-cols-3 bg-lime-300/10 px-4 py-2 text-xs uppercase tracking-[0.3em] text-lime-100">
                <span>Day</span>
                <span>Actual (mm)</span>
                <span>Predicted (mm)</span>
              </div>
              <div className="max-h-64 overflow-auto">
                {testSeries.actual.map((value, index) => (
                  <div
                    key={`test-row-${index}`}
                    className={`grid grid-cols-3 border-t border-lime-300/10 px-4 py-2 text-xs text-slate-200 ${
                      index % 2 === 0 ? "bg-lime-300/5" : "bg-white/0"
                    }`}
                  >
                    <span>{index + 1}</span>
                    <span>{value.toFixed(3)}</span>
                    <span>{testSeries.pred[index]?.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
