from pathlib import Path

import numpy as np
import pandas as pd
import torch
from io import BytesIO
import os
import re
import shutil
import subprocess
import tempfile
from threading import Event, Thread, Lock
from uuid import uuid4
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yt_dlp

DATA_FILE = Path(__file__).resolve().parents[1] / "data_rainfall.xlsx"

FEATURE_COLS = [
    "MaxAirPressure",
    "MinAirPressure",
    "AvgAirPressure8Time",
    "MaxTemp",
    "MinTemp",
    "AvgTemp",
    "Evaporation",
    "MaxHumidity",
    "MinHumidity",
    "AvgHumidity",
]

TARGET_COL = "Rainfall"


class TrainRequest(BaseModel):
    time_step: int = Field(7, ge=2, le=30)
    train_ratio: float = Field(0.8, gt=0.5, lt=0.95)
    max_epochs: int = Field(200, ge=10, le=20000)
    batch_size: int = Field(32, ge=4, le=256)
    learning_rate: float = Field(0.01, gt=0.00001, lt=0.5)


class VideoInfoRequest(BaseModel):
    url: str = Field(..., min_length=8)


class VideoDownloadRequest(BaseModel):
    url: str = Field(..., min_length=8)
    format: str = Field("mp4", pattern="^(mp4|mp3|wav)$")
    start_time: float | None = Field(None, ge=0)
    end_time: float | None = Field(None, gt=0)
    volume: float = Field(1.0, ge=0.1, le=3.0)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    fade_in: float = Field(0.0, ge=0.0, le=30.0)
    fade_out: float = Field(0.0, ge=0.0, le=30.0)


class LSTMRegressor(nn.Module):
    def __init__(self, num_features: int, hidden_units_list: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        input_size = num_features
        for units in hidden_units_list:
            self.layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=units,
                    num_layers=1,
                    batch_first=True,
                )
            )
            input_size = units
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output, _ = layer(output)
        last = output[:, -1, :]
        return self.fc(last)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")
    df = pd.read_excel(DATA_FILE)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    x = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COL].to_numpy(dtype=np.float32)
    return x, y


def make_sequences(x, y, time_step, shift=1):
    y_shifted = np.concatenate([y[shift:], np.full(shift, np.nan, dtype=np.float32)])
    valid_mask = ~np.isnan(y_shifted)
    x = x[valid_mask]
    y_shifted = y_shifted[valid_mask]

    sequences = []
    targets = []
    for i in range(time_step, x.shape[0]):
        sequences.append(x[i - time_step : i])
        targets.append(y_shifted[i])
    return np.stack(sequences), np.array(targets, dtype=np.float32)


def split_train(sequences, targets, ratio):
    num_train = int(len(targets) * ratio)
    return (
        sequences[:num_train],
        targets[:num_train],
        sequences[num_train:],
        targets[num_train:],
    )


def normalize(train_seq, test_seq):
    x_min = train_seq.min(axis=(0, 1))
    x_max = train_seq.max(axis=(0, 1))
    denom = x_max - x_min
    denom[denom == 0] = 1.0
    train_norm = (train_seq - x_min) / denom
    test_norm = (test_seq - x_min) / denom
    return train_norm, test_norm, x_min, x_max


def train_model(train_x, train_y, num_features, params: TrainRequest):
    device = torch.device("cpu")
    model = LSTMRegressor(num_features, params.hidden_units).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1),
    )
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    for _ in range(params.max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model


def predict(model, sequences):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(sequences, dtype=torch.float32)).squeeze(-1)
    preds = preds.numpy()
    preds[preds < 0] = 0
    return preds


def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE:
# This module keeps MLToolkit + video endpoints for local development.
# Render deployment uses server/render_app.py (yt-dlp only + frontend static files).


jobs = {}
jobs_lock = Lock()


def run_training(job_id, content, params: TrainRequest, hidden_units_list: list[int]):
    try:
        df = pd.read_excel(BytesIO(content))
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])
        x = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = df[TARGET_COL].to_numpy(dtype=np.float32)
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(exc)
        return

    sequences, targets = make_sequences(x, y, params.time_step)
    if len(targets) < 10:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Not enough data to train."
        return

    train_x, train_y, test_x, test_y = split_train(sequences, targets, params.train_ratio)
    train_x, test_x, x_min, x_max = normalize(train_x, test_x)

    device = torch.device("cpu")
    model = LSTMRegressor(train_x.shape[2], hidden_units_list).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1),
    )
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

    for epoch in range(params.max_epochs):
        with jobs_lock:
            if jobs[job_id]["cancel"].is_set():
                jobs[job_id]["status"] = "cancelled"
                return
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            with jobs_lock:
                jobs[job_id]["progress"] = (epoch + 1) / params.max_epochs
                jobs[job_id]["current_epoch"] = epoch + 1

    pred_train = predict(model, train_x)
    pred_test = predict(model, test_x)

    result = {
        "params": {**params.model_dump(), "hidden_units": hidden_units_list},
        "metrics": {
            "train": metrics(train_y, pred_train),
            "test": metrics(test_y, pred_test),
        },
        "series": {
            "train": {"actual": train_y.tolist(), "pred": pred_train.tolist()},
            "test": {"actual": test_y.tolist(), "pred": pred_test.tolist()},
        },
        "normalization": {
            "x_min": x_min.tolist(),
            "x_max": x_max.tolist(),
        },
    }

    with jobs_lock:
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result


@app.post("/train")
def train_endpoint(
    file: UploadFile = File(...),
    time_step: int = Form(7),
    train_ratio: float = Form(0.8),
    max_epochs: int = Form(200),
    batch_size: int = Form(32),
    learning_rate: float = Form(0.01),
    hidden_units: List[int] = Form(...),
):
    params = TrainRequest(
        time_step=time_step,
        train_ratio=train_ratio,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    hidden_units_list = [int(u) for u in hidden_units if int(u) > 0]
    if len(hidden_units_list) == 0:
        raise HTTPException(status_code=400, detail="At least one hidden unit must be > 0.")
    if len(hidden_units_list) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 layers.")
    if any(u < 0 or u > 256 for u in hidden_units_list):
        raise HTTPException(status_code=400, detail="Hidden units must be between 0 and 256.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file.")
    content = file.file.read()

    job_id = str(uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": params.max_epochs,
            "cancel": Event(),
            "result": None,
            "error": "",
        }

    thread = Thread(
        target=run_training, args=(job_id, content, params, hidden_units_list), daemon=True
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/train/{job_id}")
def train_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "status": job["status"],
            "progress": job["progress"],
            "current_epoch": job["current_epoch"],
            "total_epochs": job["total_epochs"],
            "result": job["result"],
            "error": job["error"],
        }


@app.post("/train/{job_id}/cancel")
def train_cancel(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job["cancel"].set()
        job["status"] = "cancelled"
    return {"status": "cancelled"}


def _validate_video_url(url: str):
    if not re.match(r"^https?://", url.strip()):
        raise HTTPException(status_code=400, detail="Invalid URL")


def _run_ffmpeg(command: list[str]):
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found on server. Please install ffmpeg first.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = stderr.splitlines()[-1] if stderr else "ffmpeg processing failed"
        raise HTTPException(status_code=400, detail=f"Edit failed: {detail}") from exc


def _audio_filter_chain(
    volume: float,
    speed: float,
    fade_in: float,
    fade_out: float,
    clip_duration: float | None,
):
    filters = []
    if abs(volume - 1.0) > 1e-6:
        filters.append(f"volume={volume}")
    if abs(speed - 1.0) > 1e-6:
        filters.append(f"atempo={speed}")
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0 and clip_duration and clip_duration > fade_out:
        start_out = max(0.0, clip_duration - fade_out)
        filters.append(f"afade=t=out:st={start_out}:d={fade_out}")
    return ",".join(filters)


@app.post("/video/info")
def video_info(payload: VideoInfoRequest):
    url = payload.url.strip()
    _validate_video_url(url)
    try:
        ydl_opts = {"quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch info: {exc}") from exc

    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
        "webpage_url": info.get("webpage_url"),
        "uploader": info.get("uploader"),
        "extractor": info.get("extractor_key") or info.get("extractor"),
    }


def _prepare_video_asset(payload: VideoDownloadRequest):
    url = payload.url.strip()
    _validate_video_url(url)
    file_format = payload.format.lower()
    start_time = payload.start_time
    end_time = payload.end_time
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    temp_dir = tempfile.mkdtemp(prefix="video_dl_")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
        "restrictfilenames": True,
    }

    if file_format in {"mp3", "wav"}:
        ydl_opts.update(
            {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": file_format,
                        **({"preferredquality": "192"} if file_format == "mp3" else {}),
                    }
                ],
            }
        )
    else:
        ydl_opts.update(
            {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            }
        )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            if file_format in {"mp3", "wav"}:
                base, _ = os.path.splitext(filepath)
                filepath = f"{base}.{file_format}"
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Download failed: {exc}") from exc

    if not os.path.exists(filepath):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=404, detail="Downloaded file not found")

    source_duration = float(info.get("duration")) if info.get("duration") else None
    if source_duration is not None:
        if start_time is not None and start_time >= source_duration:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="start_time exceeds source duration")
        if end_time is not None and end_time > source_duration:
            end_time = source_duration
        if start_time is not None and end_time is not None and end_time <= start_time:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="trim range is empty")

    needs_audio_edit = (
        file_format in {"mp3", "wav"}
        and (
            start_time is not None
            or end_time is not None
            or abs(payload.volume - 1.0) > 1e-6
            or abs(payload.speed - 1.0) > 1e-6
            or payload.fade_in > 0
            or payload.fade_out > 0
        )
    )
    needs_video_trim = file_format == "mp4" and (start_time is not None or end_time is not None)

    if needs_audio_edit or needs_video_trim:
        edited_path = os.path.join(temp_dir, f"edited.{file_format}")
        ffmpeg_cmd = ["ffmpeg", "-y"]
        if start_time is not None:
            ffmpeg_cmd.extend(["-ss", str(start_time)])
        ffmpeg_cmd.extend(["-i", filepath])
        if end_time is not None:
            if start_time is not None:
                ffmpeg_cmd.extend(["-t", str(end_time - start_time)])
            else:
                ffmpeg_cmd.extend(["-to", str(end_time)])

        if file_format == "mp4":
            ffmpeg_cmd.extend(["-c", "copy", edited_path])
        else:
            clip_duration = None
            if source_duration is not None:
                clip_start = start_time or 0.0
                clip_end = end_time if end_time is not None else source_duration
                clip_duration = max(0.0, clip_end - clip_start)
                if payload.fade_in + payload.fade_out >= clip_duration > 0:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=400,
                        detail="fade_in + fade_out must be shorter than clip duration",
                    )
            filters = _audio_filter_chain(
                volume=payload.volume,
                speed=payload.speed,
                fade_in=payload.fade_in,
                fade_out=payload.fade_out,
                clip_duration=clip_duration,
            )
            if filters:
                ffmpeg_cmd.extend(["-af", filters])
            if file_format == "mp3":
                ffmpeg_cmd.extend(["-vn", "-c:a", "libmp3lame", "-b:a", "192k", edited_path])
            else:
                ffmpeg_cmd.extend(["-vn", "-c:a", "pcm_s16le", edited_path])

        _run_ffmpeg(ffmpeg_cmd)
        filepath = edited_path

    return filepath, temp_dir, file_format


@app.post("/video/download")
def video_download(payload: VideoDownloadRequest, background_tasks: BackgroundTasks):
    filepath, temp_dir, _ = _prepare_video_asset(payload)
    background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
    filename = os.path.basename(filepath)
    return FileResponse(filepath, filename=filename, media_type="application/octet-stream")


@app.post("/video/preview")
def video_preview(payload: VideoDownloadRequest, background_tasks: BackgroundTasks):
    filepath, temp_dir, file_format = _prepare_video_asset(payload)
    background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
    media_type = {
        "mp4": "video/mp4",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
    }.get(file_format, "application/octet-stream")
    return FileResponse(filepath, media_type=media_type)
