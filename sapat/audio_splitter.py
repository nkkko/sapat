import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple


def check_ffmpeg_availability():
    """Check if ffmpeg and ffprobe are available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)


def get_audio_info(filepath: str) -> Tuple[float, int]:
    """
    Get audio duration and bitrate using ffprobe
    
    Returns:
        Tuple of (duration_seconds, bitrate_bps)
    """
    # Get duration
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', filepath
    ]
    
    try:
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        raise RuntimeError(f"Failed to get audio duration for {filepath}")
    
    # Get bitrate
    bitrate_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=bit_rate', '-of', 'json', filepath
    ]
    
    try:
        result = subprocess.run(bitrate_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        bitrate = int(data['streams'][0]['bit_rate'])
    except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError):
        # Fallback: calculate bitrate from file size and duration
        file_size_bytes = os.path.getsize(filepath)
        bitrate = int((file_size_bytes * 8) / duration)
    
    return duration, bitrate


def calculate_segment_duration(bitrate_bps: int, max_size_mb: float = 24.0) -> float:
    """
    Calculate segment duration to achieve target file size
    
    Args:
        bitrate_bps: Bitrate in bits per second
        max_size_mb: Maximum size per segment in MB (default 24MB for safety)
        
    Returns:
        Duration in seconds for each segment
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    max_size_bits = max_size_bytes * 8
    return max_size_bits / bitrate_bps


def split_audio_file(input_file: str, max_size_mb: float = 24.0) -> List[str]:
    """
    Split audio file into chunks of approximately max_size_mb
    
    Args:
        input_file: Path to the input audio file
        max_size_mb: Maximum size per chunk in MB
        
    Returns:
        List of paths to the created chunk files
    """
    # Check if ffmpeg tools are available
    if not check_ffmpeg_availability():
        raise RuntimeError("ffmpeg and ffprobe are required for splitting large audio files. Please install FFmpeg.")
    
    input_path = Path(input_file)
    
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp(prefix=f"sapat_chunks_{input_path.stem}_")
    
    try:
        # Get audio information
        duration, bitrate = get_audio_info(input_file)
        
        # Calculate segment duration
        segment_duration = calculate_segment_duration(bitrate, max_size_mb)
        
        # Ensure minimum segment duration to avoid too many small chunks
        segment_duration = max(segment_duration, 30.0)  # At least 30 seconds
        
        # Output pattern for chunks
        output_pattern = os.path.join(temp_dir, f"chunk_%03d{input_path.suffix}")
        
        # FFmpeg command to split the file
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-f', 'segment',
            '-segment_time', str(segment_duration),
            '-c', 'copy',  # Stream copy to avoid re-encoding
            '-reset_timestamps', '1',
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output files
            output_pattern
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg splitting failed: {result.stderr}")
        
        # Get list of created chunk files
        chunk_files = sorted([
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.startswith("chunk_")
        ])
        
        if not chunk_files:
            raise RuntimeError("No chunk files were created")
        
        return chunk_files
        
    except Exception as e:
        # Clean up on error
        cleanup_chunks([temp_dir])
        raise e


def cleanup_chunks(chunk_paths: List[str]):
    """
    Clean up temporary chunk files and directories
    
    Args:
        chunk_paths: List of file or directory paths to clean up
    """
    for path in chunk_paths:
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                path_obj.unlink()
            elif path_obj.is_dir():
                # Remove all files in directory first
                for file_path in path_obj.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                # Remove the directory
                path_obj.rmdir()
        except (OSError, FileNotFoundError):
            # Ignore cleanup errors
            pass


def should_split_file(filepath: str, max_size_mb: float = 25.0) -> bool:
    """
    Check if file should be split based on size
    
    Args:
        filepath: Path to the file to check
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if file should be split, False otherwise
    """
    try:
        return get_file_size_mb(filepath) > max_size_mb
    except OSError:
        return False