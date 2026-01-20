from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
import subprocess
import tempfile
import uuid
import os
import requests
import shutil

app = FastAPI(title="Video Generator API", version="1.0.0")

# Storage directory for generated videos
VIDEOS_DIR = "/app/videos"
os.makedirs(VIDEOS_DIR, exist_ok=True)

class VideoRequest(BaseModel):
    audio_url: str  # URL to audio file (from ElevenLabs)
    images: List[str]  # List of image URLs
    title: Optional[str] = "Video"
    output_format: str = "mp4"
    resolution: str = "1920x1080"

class TextOverlayRequest(BaseModel):
    audio_url: str
    text_segments: List[dict]  # [{"text": "...", "duration": 5}, ...]
    background_color: str = "black"
    font_color: str = "white"
    font_size: int = 48
    resolution: str = "1920x1080"

class VideoResponse(BaseModel):
    video_id: str
    status: str
    video_url: Optional[str] = None

def download_file(url: str, dest_path: str):
    """Download a file from URL"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "video-generator"}

@app.post("/generate/slideshow", response_model=VideoResponse)
async def generate_slideshow(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate a slideshow video from images with audio"""
    video_id = str(uuid.uuid4())

    try:
        work_dir = tempfile.mkdtemp()

        # Download audio
        audio_path = os.path.join(work_dir, "audio.mp3")
        download_file(request.audio_url, audio_path)

        # Get audio duration
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ], capture_output=True, text=True)
        audio_duration = float(result.stdout.strip())

        # Download images
        image_duration = audio_duration / len(request.images)
        image_paths = []
        for i, img_url in enumerate(request.images):
            img_path = os.path.join(work_dir, f"image_{i}.jpg")
            download_file(img_url, img_path)
            image_paths.append(img_path)

        # Create video from images
        output_path = os.path.join(VIDEOS_DIR, f"{video_id}.{request.output_format}")

        # Create concat file
        concat_file = os.path.join(work_dir, "concat.txt")
        with open(concat_file, 'w') as f:
            for img_path in image_paths:
                f.write(f"file '{img_path}'\n")
                f.write(f"duration {image_duration}\n")
            f.write(f"file '{image_paths[-1]}'\n")

        # Generate video with FFmpeg
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", audio_path,
            "-vf", f"scale={request.resolution}:force_original_aspect_ratio=decrease,pad={request.resolution}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path
        ], check=True)

        # Cleanup
        shutil.rmtree(work_dir)

        return VideoResponse(
            video_id=video_id,
            status="completed",
            video_url=f"/videos/{video_id}.{request.output_format}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/text-video", response_model=VideoResponse)
async def generate_text_video(request: TextOverlayRequest):
    """Generate a video with text overlays and audio"""
    video_id = str(uuid.uuid4())

    try:
        work_dir = tempfile.mkdtemp()

        # Download audio
        audio_path = os.path.join(work_dir, "audio.mp3")
        download_file(request.audio_url, audio_path)

        # Create filter complex for text segments
        filter_parts = []
        current_time = 0

        for i, segment in enumerate(request.text_segments):
            text = segment["text"].replace("'", "\\'").replace(":", "\\:")
            duration = segment.get("duration", 5)
            end_time = current_time + duration

            filter_parts.append(
                f"drawtext=text='{text}':fontcolor={request.font_color}:fontsize={request.font_size}:"
                f"x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,{current_time},{end_time})'"
            )
            current_time = end_time

        filter_complex = ",".join(filter_parts)

        output_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")

        # Generate video
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={request.background_color}:s={request.resolution}:d={current_time}",
            "-i", audio_path,
            "-vf", filter_complex,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path
        ], check=True)

        shutil.rmtree(work_dir)

        return VideoResponse(
            video_id=video_id,
            status="completed",
            video_url=f"/videos/{video_id}.mp4"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos/{video_id}")
async def get_video(video_id: str):
    """Download generated video"""
    video_path = os.path.join(VIDEOS_DIR, video_id)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a generated video"""
    video_path = os.path.join(VIDEOS_DIR, video_id)
    if os.path.exists(video_path):
        os.remove(video_path)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Video not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
