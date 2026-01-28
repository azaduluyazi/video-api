"""
Troia Media Video Generator API v2.0
Full-featured FFmpeg-based video generation with templates, subtitles, and progress tracking.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import subprocess
import tempfile
import uuid
import os
import json
import requests
import shutil
import asyncio
from datetime import datetime
from enum import Enum

app = FastAPI(title="Troia Video Generator API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage directories
VIDEOS_DIR = "/app/videos"
THUMBNAILS_DIR = "/app/thumbnails"
TEMP_DIR = "/app/temp"

for d in [VIDEOS_DIR, THUMBNAILS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# Job tracking
jobs: Dict[str, Dict] = {}

# Dashboard webhook URL
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://a4swwg8sogsssow44c0g4k4c.troiamedia.cloud")


class VideoStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoTemplate(str, Enum):
    SLIDESHOW = "slideshow"
    TEXT_OVERLAY = "text_overlay"
    NEWS_STYLE = "news_style"
    DOCUMENTARY = "documentary"
    SHORTS = "shorts"


# ==================== REQUEST MODELS ====================

class SubtitleSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    style: Optional[str] = "default"


class VideoRequest(BaseModel):
    audio_url: str
    images: List[str]
    title: Optional[str] = "Video"
    description: Optional[str] = ""
    template: Optional[VideoTemplate] = VideoTemplate.SLIDESHOW
    resolution: str = "1920x1080"
    fps: int = 30
    subtitles: Optional[List[SubtitleSegment]] = None
    generate_thumbnail: bool = True
    thumbnail_time: float = 5.0
    webhook_url: Optional[str] = None
    transition: Optional[str] = "fade"  # fade, slide, zoom, none
    transition_duration: float = 0.5
    background_music_volume: float = 1.0
    text_overlay: Optional[Dict[str, Any]] = None


class TextVideoRequest(BaseModel):
    audio_url: str
    text_segments: List[Dict[str, Any]]
    background_color: str = "#000000"
    background_image: Optional[str] = None
    font_color: str = "#FFFFFF"
    font_size: int = 48
    font_family: str = "Arial"
    resolution: str = "1920x1080"
    subtitles: Optional[List[SubtitleSegment]] = None
    generate_thumbnail: bool = True
    webhook_url: Optional[str] = None


class ShortsRequest(BaseModel):
    audio_url: str
    images: List[str]
    title: str
    captions: Optional[List[SubtitleSegment]] = None
    resolution: str = "1080x1920"  # Vertical for shorts
    generate_thumbnail: bool = True
    webhook_url: Optional[str] = None


class DocumentaryRequest(BaseModel):
    audio_url: str
    script_segments: List[Dict[str, Any]]  # {text, images, duration}
    title: str
    intro_text: Optional[str] = None
    outro_text: Optional[str] = None
    resolution: str = "1920x1080"
    generate_thumbnail: bool = True
    webhook_url: Optional[str] = None


# ==================== HELPER FUNCTIONS ====================

def download_file(url: str, dest_path: str) -> bool:
    """Download a file from URL"""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())


def generate_subtitle_file(subtitles: List[SubtitleSegment], output_path: str) -> str:
    """Generate SRT subtitle file"""
    srt_content = ""
    for i, sub in enumerate(subtitles, 1):
        start = format_srt_time(sub.start_time)
        end = format_srt_time(sub.end_time)
        srt_content += f"{i}\n{start} --> {end}\n{sub.text}\n\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    return output_path


def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_thumbnail(video_path: str, output_path: str, time: float = 5.0) -> bool:
    """Generate thumbnail from video"""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(time), "-i", video_path,
            "-vframes", "1", "-q:v", "2", output_path
        ], check=True, capture_output=True)
        return True
    except:
        return False


async def notify_webhook(url: str, data: Dict):
    """Send webhook notification"""
    try:
        requests.post(url, json=data, timeout=10)
    except:
        pass


async def notify_dashboard(event_type: str, message: str, details: Dict = None):
    """Send event to dashboard"""
    try:
        requests.post(f"{DASHBOARD_URL}/api/events/log", json={
            "type": event_type,
            "source": "video-api",
            "message": message,
            "details": details or {}
        }, timeout=5)
    except:
        pass


async def update_pipeline(stage: str, progress: int, video_info: Dict = None):
    """Update pipeline status on dashboard"""
    try:
        requests.post(f"{DASHBOARD_URL}/api/pipeline/update", json={
            "stage": stage,
            "progress": progress,
            "video": video_info
        }, timeout=5)
    except:
        pass


def update_job_status(job_id: str, status: VideoStatus, progress: int = 0,
                      message: str = "", result: Dict = None):
    """Update job status"""
    if job_id in jobs:
        jobs[job_id].update({
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": datetime.utcnow().isoformat()
        })
        if result:
            jobs[job_id]["result"] = result


# ==================== VIDEO GENERATION FUNCTIONS ====================

async def generate_slideshow_video(job_id: str, request: VideoRequest):
    """Generate slideshow video with transitions"""
    work_dir = tempfile.mkdtemp(dir=TEMP_DIR)

    try:
        update_job_status(job_id, VideoStatus.PROCESSING, 10, "Downloading audio...")
        await notify_dashboard("info", f"Starting video generation: {request.title}")
        await update_pipeline("generate", 10, {"title": request.title, "job_id": job_id})

        # Download audio
        audio_path = os.path.join(work_dir, "audio.mp3")
        if not download_file(request.audio_url, audio_path):
            raise Exception("Failed to download audio")

        audio_duration = get_audio_duration(audio_path)
        update_job_status(job_id, VideoStatus.PROCESSING, 20, "Downloading images...")

        # Download images
        image_paths = []
        for i, img_url in enumerate(request.images):
            img_path = os.path.join(work_dir, f"image_{i:03d}.jpg")
            if download_file(img_url, img_path):
                image_paths.append(img_path)

        if not image_paths:
            raise Exception("No images downloaded successfully")

        update_job_status(job_id, VideoStatus.RENDERING, 40, "Rendering video...")
        await update_pipeline("render", 40, {"title": request.title})

        # Calculate image duration
        image_duration = audio_duration / len(image_paths)

        # Create video filter for transitions
        width, height = request.resolution.split('x')

        # Build FFmpeg command based on transition type
        output_path = os.path.join(VIDEOS_DIR, f"{job_id}.mp4")

        if request.transition == "fade":
            # Create concat file with crossfade
            filter_complex = []
            inputs = []

            for i, img_path in enumerate(image_paths):
                inputs.extend(["-loop", "1", "-t", str(image_duration), "-i", img_path])

            inputs.extend(["-i", audio_path])

            # Build xfade filter chain
            if len(image_paths) > 1:
                td = request.transition_duration
                for i in range(len(image_paths) - 1):
                    if i == 0:
                        filter_complex.append(f"[0][1]xfade=transition=fade:duration={td}:offset={image_duration - td}[v1]")
                    else:
                        offset = (i + 1) * image_duration - td * (i + 1)
                        filter_complex.append(f"[v{i}][{i+1}]xfade=transition=fade:duration={td}:offset={offset}[v{i+1}]")

                last_v = f"v{len(image_paths)-1}"
                filter_str = ";".join(filter_complex) + f";[{last_v}]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2[vout]"
            else:
                filter_str = f"[0]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2[vout]"

            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_str,
                "-map", "[vout]", "-map", f"{len(image_paths)}:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", output_path
            ]
        else:
            # Simple concat without transitions
            concat_file = os.path.join(work_dir, "concat.txt")
            with open(concat_file, 'w') as f:
                for img_path in image_paths:
                    f.write(f"file '{img_path}'\n")
                    f.write(f"duration {image_duration}\n")
                f.write(f"file '{image_paths[-1]}'\n")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-i", audio_path,
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", output_path
            ]

        subprocess.run(cmd, check=True, capture_output=True)
        update_job_status(job_id, VideoStatus.RENDERING, 70, "Adding subtitles...")

        # Add subtitles if provided
        if request.subtitles:
            srt_path = os.path.join(work_dir, "subtitles.srt")
            generate_subtitle_file(request.subtitles, srt_path)

            temp_output = os.path.join(work_dir, "temp_output.mp4")
            shutil.move(output_path, temp_output)

            subprocess.run([
                "ffmpeg", "-y", "-i", temp_output,
                "-vf", f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF&'",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy", output_path
            ], check=True, capture_output=True)

        update_job_status(job_id, VideoStatus.RENDERING, 90, "Generating thumbnail...")

        # Generate thumbnail
        thumbnail_path = None
        if request.generate_thumbnail:
            thumbnail_path = os.path.join(THUMBNAILS_DIR, f"{job_id}.jpg")
            generate_thumbnail(output_path, thumbnail_path, request.thumbnail_time)

        # Complete
        result = {
            "video_id": job_id,
            "video_url": f"/videos/{job_id}.mp4",
            "thumbnail_url": f"/thumbnails/{job_id}.jpg" if thumbnail_path else None,
            "duration": audio_duration,
            "resolution": request.resolution
        }

        update_job_status(job_id, VideoStatus.COMPLETED, 100, "Video ready!", result)
        await notify_dashboard("success", f"Video completed: {request.title}", result)
        await update_pipeline("upload", 100, {"title": request.title, "completed": True})

        # Webhook notification
        if request.webhook_url:
            await notify_webhook(request.webhook_url, {
                "event": "video_completed",
                "job_id": job_id,
                "result": result
            })

    except Exception as e:
        error_msg = str(e)
        update_job_status(job_id, VideoStatus.FAILED, 0, error_msg)
        await notify_dashboard("error", f"Video generation failed: {error_msg}")

        if request.webhook_url:
            await notify_webhook(request.webhook_url, {
                "event": "video_failed",
                "job_id": job_id,
                "error": error_msg
            })

    finally:
        # Cleanup
        try:
            shutil.rmtree(work_dir)
        except:
            pass


async def generate_text_video(job_id: str, request: TextVideoRequest):
    """Generate text overlay video"""
    work_dir = tempfile.mkdtemp(dir=TEMP_DIR)

    try:
        update_job_status(job_id, VideoStatus.PROCESSING, 10, "Preparing...")

        # Download audio
        audio_path = os.path.join(work_dir, "audio.mp3")
        download_file(request.audio_url, audio_path)
        audio_duration = get_audio_duration(audio_path)

        update_job_status(job_id, VideoStatus.RENDERING, 30, "Rendering...")

        # Build text filter
        width, height = request.resolution.split('x')
        filter_parts = []
        current_time = 0

        for segment in request.text_segments:
            text = segment["text"].replace("'", "\\'").replace(":", "\\:")
            duration = segment.get("duration", 5)
            end_time = current_time + duration

            filter_parts.append(
                f"drawtext=text='{text}':fontcolor={request.font_color}:fontsize={request.font_size}:"
                f"x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,{current_time},{end_time})'"
            )
            current_time = end_time

        filter_complex = ",".join(filter_parts)
        output_path = os.path.join(VIDEOS_DIR, f"{job_id}.mp4")

        # Handle background
        if request.background_image:
            bg_path = os.path.join(work_dir, "background.jpg")
            download_file(request.background_image, bg_path)
            input_args = ["-loop", "1", "-i", bg_path]
        else:
            bg_color = request.background_color.replace("#", "")
            input_args = ["-f", "lavfi", "-i", f"color=c=0x{bg_color}:s={request.resolution}:d={current_time}"]

        cmd = ["ffmpeg", "-y"] + input_args + [
            "-i", audio_path,
            "-vf", filter_complex,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(current_time),
            "-shortest", output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        # Thumbnail
        thumbnail_path = None
        if request.generate_thumbnail:
            thumbnail_path = os.path.join(THUMBNAILS_DIR, f"{job_id}.jpg")
            generate_thumbnail(output_path, thumbnail_path)

        result = {
            "video_id": job_id,
            "video_url": f"/videos/{job_id}.mp4",
            "thumbnail_url": f"/thumbnails/{job_id}.jpg" if thumbnail_path else None,
            "duration": min(audio_duration, current_time)
        }

        update_job_status(job_id, VideoStatus.COMPLETED, 100, "Ready!", result)
        await notify_dashboard("success", f"Text video completed: {job_id}")

    except Exception as e:
        update_job_status(job_id, VideoStatus.FAILED, 0, str(e))
        await notify_dashboard("error", f"Text video failed: {str(e)}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


async def generate_shorts_video(job_id: str, request: ShortsRequest):
    """Generate vertical shorts video with captions"""
    work_dir = tempfile.mkdtemp(dir=TEMP_DIR)

    try:
        update_job_status(job_id, VideoStatus.PROCESSING, 10, "Preparing shorts...")
        await notify_dashboard("info", f"Generating shorts: {request.title}")

        # Download audio
        audio_path = os.path.join(work_dir, "audio.mp3")
        download_file(request.audio_url, audio_path)
        audio_duration = get_audio_duration(audio_path)

        # Download images
        image_paths = []
        for i, url in enumerate(request.images):
            path = os.path.join(work_dir, f"img_{i}.jpg")
            if download_file(url, path):
                image_paths.append(path)

        update_job_status(job_id, VideoStatus.RENDERING, 40, "Rendering shorts...")

        # Create vertical video
        width, height = request.resolution.split('x')
        image_duration = audio_duration / max(len(image_paths), 1)

        concat_file = os.path.join(work_dir, "concat.txt")
        with open(concat_file, 'w') as f:
            for img in image_paths:
                f.write(f"file '{img}'\nduration {image_duration}\n")
            if image_paths:
                f.write(f"file '{image_paths[-1]}'\n")

        output_path = os.path.join(VIDEOS_DIR, f"{job_id}.mp4")

        # Video filter for vertical format with zoom effect
        vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},zoompan=z='min(zoom+0.001,1.3)':d={int(image_duration*30)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}"

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", audio_path,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        # Add captions if provided
        if request.captions:
            srt_path = os.path.join(work_dir, "captions.srt")
            generate_subtitle_file(request.captions, srt_path)

            temp_out = os.path.join(work_dir, "temp.mp4")
            shutil.move(output_path, temp_out)

            # Shorts-style captions (centered, larger)
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_out,
                "-vf", f"subtitles={srt_path}:force_style='FontSize=32,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,Outline=2,Alignment=2'",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy", output_path
            ], check=True, capture_output=True)

        # Thumbnail
        thumbnail_path = os.path.join(THUMBNAILS_DIR, f"{job_id}.jpg")
        generate_thumbnail(output_path, thumbnail_path, 1.0)

        result = {
            "video_id": job_id,
            "video_url": f"/videos/{job_id}.mp4",
            "thumbnail_url": f"/thumbnails/{job_id}.jpg",
            "duration": audio_duration,
            "format": "shorts"
        }

        update_job_status(job_id, VideoStatus.COMPLETED, 100, "Shorts ready!", result)
        await notify_dashboard("success", f"Shorts completed: {request.title}")

    except Exception as e:
        update_job_status(job_id, VideoStatus.FAILED, 0, str(e))
        await notify_dashboard("error", f"Shorts generation failed: {str(e)}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ==================== API ENDPOINTS ====================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "troia-video-generator",
        "version": "2.0.0",
        "active_jobs": len([j for j in jobs.values() if j["status"] in [VideoStatus.QUEUED, VideoStatus.PROCESSING, VideoStatus.RENDERING]])
    }


@app.post("/api/videos/slideshow")
async def create_slideshow(request: VideoRequest, background_tasks: BackgroundTasks):
    """Create slideshow video from images with audio"""
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "id": job_id,
        "type": "slideshow",
        "status": VideoStatus.QUEUED,
        "progress": 0,
        "message": "Queued",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.dict()
    }

    background_tasks.add_task(generate_slideshow_video, job_id, request)

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/videos/{job_id}/status"
    }


@app.post("/api/videos/text")
async def create_text_video(request: TextVideoRequest, background_tasks: BackgroundTasks):
    """Create text overlay video"""
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "id": job_id,
        "type": "text_overlay",
        "status": VideoStatus.QUEUED,
        "progress": 0,
        "created_at": datetime.utcnow().isoformat()
    }

    background_tasks.add_task(generate_text_video, job_id, request)

    return {"job_id": job_id, "status": "queued"}


@app.post("/api/videos/shorts")
async def create_shorts(request: ShortsRequest, background_tasks: BackgroundTasks):
    """Create vertical shorts video"""
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "id": job_id,
        "type": "shorts",
        "status": VideoStatus.QUEUED,
        "progress": 0,
        "created_at": datetime.utcnow().isoformat()
    }

    background_tasks.add_task(generate_shorts_video, job_id, request)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/videos/{job_id}/status")
async def get_job_status(job_id: str):
    """Get video generation job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "result": job.get("result"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at")
    }


@app.get("/api/videos")
async def list_jobs(limit: int = 20):
    """List recent video jobs"""
    sorted_jobs = sorted(jobs.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return {"jobs": sorted_jobs[:limit], "total": len(jobs)}


@app.get("/videos/{filename}")
async def get_video(filename: str):
    """Download generated video"""
    video_path = os.path.join(VIDEOS_DIR, filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


@app.get("/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    """Get video thumbnail"""
    thumb_path = os.path.join(THUMBNAILS_DIR, filename)
    if not os.path.exists(thumb_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumb_path, media_type="image/jpeg")


@app.delete("/api/videos/{job_id}")
async def delete_video(job_id: str):
    """Delete video and thumbnail"""
    video_path = os.path.join(VIDEOS_DIR, f"{job_id}.mp4")
    thumb_path = os.path.join(THUMBNAILS_DIR, f"{job_id}.jpg")

    deleted = []
    if os.path.exists(video_path):
        os.remove(video_path)
        deleted.append("video")
    if os.path.exists(thumb_path):
        os.remove(thumb_path)
        deleted.append("thumbnail")

    if job_id in jobs:
        del jobs[job_id]
        deleted.append("job")

    return {"deleted": deleted}


# ==================== LEGACY ENDPOINTS (backward compatible) ====================

@app.post("/generate/slideshow")
async def legacy_slideshow(request: VideoRequest, background_tasks: BackgroundTasks):
    """Legacy endpoint for backward compatibility"""
    return await create_slideshow(request, background_tasks)


@app.post("/generate/text-video")
async def legacy_text_video(request: TextVideoRequest, background_tasks: BackgroundTasks):
    """Legacy endpoint for backward compatibility"""
    return await create_text_video(request, background_tasks)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
