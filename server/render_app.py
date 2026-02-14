import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yt_dlp
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = BASE_DIR / "dist"
INDEX_FILE = DIST_DIR / "index.html"


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


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/api/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/api/video/info")
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
        ydl_opts.update({"format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"})

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


@app.post("/api/video/download")
def video_download(payload: VideoDownloadRequest, background_tasks: BackgroundTasks):
    filepath, temp_dir, _ = _prepare_video_asset(payload)
    background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
    filename = os.path.basename(filepath)
    return FileResponse(filepath, filename=filename, media_type="application/octet-stream")


@app.post("/api/video/preview")
def video_preview(payload: VideoDownloadRequest, background_tasks: BackgroundTasks):
    filepath, temp_dir, file_format = _prepare_video_asset(payload)
    background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
    media_type = {
        "mp4": "video/mp4",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
    }.get(file_format, "application/octet-stream")
    return FileResponse(filepath, media_type=media_type)


def _safe_dist_path(full_path: str) -> Path | None:
    if not DIST_DIR.exists():
        return None
    resolved = (DIST_DIR / full_path).resolve()
    try:
        resolved.relative_to(DIST_DIR.resolve())
    except ValueError:
        return None
    if resolved.is_file():
        return resolved
    return None


@app.get("/", include_in_schema=False)
def serve_root():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Frontend build not found in /dist.")
    return FileResponse(INDEX_FILE)


@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    target = _safe_dist_path(full_path)
    if target:
        return FileResponse(target)

    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Frontend build not found in /dist.")
    return FileResponse(INDEX_FILE)
