import os
import subprocess
import sys

def download_youtube_clip(url, start_time, end_time, output_path="samples/youtube_drone_fire.mp4"):
    print(f"[YouTube] Downloading clip from {url} ({start_time} to {end_time}) in 4K...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    command = [
        ".venv/bin/yt-dlp",
        "-f", "bestvideo[height<=2160]+bestaudio/best[height<=2160]",
        "--download-sections", f"*{start_time}-{end_time}",
        "--force-keyframes-at-cuts",
        url,
        "-o", output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"[YouTube] Successfully downloaded video to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[YouTube] Error downloading video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    url = "https://youtu.be/9mb3O5t_f3M"
    download_youtube_clip(url, "00:30", "01:50", "samples/youtube_drone_fire.mp4")
