# ABOUTME: Audio format conversion utilities using ffmpeg
# ABOUTME: Supports MP3 (with quality levels), WAV, FLAC, and OGG output

import subprocess

from sapat.providers.base import AudioFormat


def convert_audio(input_file: str, output_file: str, quality: str, target_format: AudioFormat = AudioFormat.MP3):
    """Convert video/audio to the specified format using ffmpeg."""
    if target_format == AudioFormat.MP3:
        _convert_to_mp3(input_file, output_file, quality)
    elif target_format == AudioFormat.WAV:
        _convert_to_wav(input_file, output_file)
    elif target_format == AudioFormat.FLAC:
        _convert_to_flac(input_file, output_file)
    elif target_format == AudioFormat.OGG:
        _convert_to_ogg(input_file, output_file)
    else:
        raise ValueError(f"Unsupported target format: {target_format}")


def _convert_to_mp3(input_file: str, output_file: str, quality: str):
    """Convert to MP3 with quality presets."""
    if quality == "L":
        opts = ["-ar", "22050", "-ac", "1", "-b:a", "96k"]
    elif quality == "M":
        opts = ["-ar", "44100", "-ac", "1", "-b:a", "96k"]
    elif quality == "H":
        opts = ["-ar", "44100", "-ac", "2", "-b:a", "160k"]
    else:
        raise ValueError("Invalid quality. Choose from 'L', 'M', 'H'.")

    cmd = ["ffmpeg", "-i", input_file, "-vn"] + opts + ["-loglevel", "error", "-y", output_file]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _convert_to_wav(input_file: str, output_file: str):
    """Convert to 16kHz mono WAV (standard for speech recognition)."""
    cmd = [
        "ffmpeg", "-i", input_file,
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
        "-loglevel", "error", "-y", output_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _convert_to_flac(input_file: str, output_file: str):
    """Convert to FLAC."""
    cmd = [
        "ffmpeg", "-i", input_file,
        "-ar", "16000", "-ac", "1",
        "-loglevel", "error", "-y", output_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _convert_to_ogg(input_file: str, output_file: str):
    """Convert to Ogg Opus."""
    cmd = [
        "ffmpeg", "-i", input_file,
        "-c:a", "libopus", "-b:a", "64k", "-ar", "16000", "-ac", "1",
        "-loglevel", "error", "-y", output_file,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
