#!/usr/bin/env python3
"""
frames_to_video_with_audio.py

Create a video from a numbered image sequence whose numbering may not start at 0
(e.g., 0749.png, 0750.png, ...), and mux in audio from a source video.

By default, audio is taken from the source video (2nd input) and is time-offset by:
    offset_seconds = start_frame_number / fps
so that frame N lines up with audio at t = N/fps.

Requires: ffmpeg on PATH.
"""

import argparse
import re
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def find_first_frame(frames_dir: Path, ext: str) -> int:
    """
    Find the smallest integer filename like 0749.png in frames_dir.
    """
    ext_clean = ext.lstrip(".")
    pat = re.compile(rf"^(\d+)\.{re.escape(ext_clean)}$", re.IGNORECASE)

    nums: list[int] = []
    for p in frames_dir.iterdir():
        if p.is_file():
            m = pat.match(p.name)
            if m:
                nums.append(int(m.group(1)))

    if not nums:
        raise ValueError(f"No numeric frame files like ####.{ext_clean} found in {frames_dir}")

    return min(nums)


def normalize_audio_map(audio_track: str, source_input_index: int = 1) -> str:
    """
    Accepts:
      - "a" or "a:0" or "a:1"  (assumed to refer to source_input_index)
      - "1:a:0"                (explicit input index, used as-is)
      - "0:a:0"                (explicit input index, used as-is)
    Returns a valid ffmpeg -map spec.
    """
    at = audio_track.strip()

    # Common shorthand: "a" means "a:0"
    if at == "a":
        at = "a:0"

    parts = at.split(":")
    if len(parts) == 1:
        # e.g. "a0" (unlikely) - treat as stream spec on source input
        return f"{source_input_index}:{at}"
    if len(parts) == 2:
        # e.g. "a:0" -> assume it belongs to source input (input 1 by default)
        return f"{source_input_index}:{at}"
    # len(parts) >= 3 e.g. "1:a:0" -> use as-is
    return at


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make a video from numbered frames and add audio from a source video."
    )
    ap.add_argument("--frames-dir", required=True, help="Directory containing frames, e.g. ./frames")
    ap.add_argument(
        "--pattern",
        default="%04d.png",
        help="ffmpeg frame pattern (default: %04d.png). Must match your filenames.",
    )
    ap.add_argument(
        "--ext",
        default=".png",
        help="Frame extension for auto-detect (default: .png). Only used when --start-number is not provided.",
    )
    ap.add_argument("--fps", type=float, required=True, help="Frames per second (e.g. 30 or 50)")
    ap.add_argument("--source-video", required=True, help="Original video file to take audio from")
    ap.add_argument("--output", required=True, help="Output video path (e.g. out.mp4)")
    ap.add_argument(
        "--start-number",
        type=int,
        default=None,
        help="Explicitly set first frame number. If omitted, auto-detected from filenames.",
    )
    ap.add_argument(
        "--no-audio-offset",
        action="store_true",
        help="Do NOT offset audio by start_number/fps (audio starts at 0).",
    )
    ap.add_argument(
        "--audio-track",
        default="a:0",
        help='Audio stream selector from source video. Examples: "a:0", "a", "1:a:0" (default: a:0).',
    )
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF quality (default: 18)")
    ap.add_argument("--preset", default="medium", help="x264 preset (default: medium)")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir).expanduser()
    source_video = Path(args.source_video).expanduser()
    output = Path(args.output).expanduser()

    if not frames_dir.exists():
        raise SystemExit(f"frames-dir not found: {frames_dir}")
    if not source_video.exists():
        raise SystemExit(f"source-video not found: {source_video}")

    start_num = args.start_number
    if start_num is None:
        start_num = find_first_frame(frames_dir, args.ext)

    # Offset audio so it aligns with the first frame index in the sequence.
    # Example: first frame 500 at 50fps => offset = 10.0 seconds
    audio_offset = 0.0 if args.no_audio_offset else (start_num / args.fps)

    frames_input = str((frames_dir / args.pattern))

    # Inputs:
    #   input 0: image sequence -> video
    #   input 1: source video   -> audio (and video, but we only map audio)
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "info",
        "-framerate",
        str(args.fps),
        "-start_number",
        str(start_num),
        "-i",
        frames_input,
    ]

    # Add source audio input with optional seek
    if audio_offset > 0:
        # fast seek (good enough for most use); move after "-i source_video" for sample-accurate seeking
        cmd += ["-ss", f"{audio_offset:.9f}"]

    cmd += ["-i", str(source_video)]

    # Stream mapping
    video_map = "0:v:0"
    audio_map = normalize_audio_map(args.audio_track, source_input_index=1)

    cmd += [
        "-map",
        video_map,
        "-map",
        audio_map,
        # Video encode
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        # Audio encode
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        # End when the shorter stream ends (usually the frames video)
        "-shortest",
        str(output),
    ]

    try:
        run(cmd)
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found. Install ffmpeg and ensure it's on your PATH.")


if __name__ == "__main__":
    main()
