import os
import subprocess
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

# Add an extra blank line
def convert_to_mp3(input_file, output_file):
    """Convert video to MP3 using ffmpeg"""
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vn',
        '-ar', '44100',
        '-ac', '2',
        '-b:a', '192k',
        output_file
    ]
    subprocess.run(command, check=True)


def transcribe_audio(audio_file):
    """Transcribe audio using Azure OpenAI Whisper API"""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/audio/transcriptions?api-version=2024-02-01"
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
    }
    with open(audio_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.json()['text']
    else:
        raise Exception(f"Transcription failed: {response.text}")

def process_file(input_file):
    """Process a single video file"""
    input_path = Path(input_file)
    mp3_file = input_path.with_suffix('.mp3')
    txt_file = input_path.with_suffix('.txt')

    print(f"Processing {input_file}")

    # Convert to MP3
    convert_to_mp3(str(input_path), str(mp3_file))
    print("Conversion to MP3 completed")

    # Transcribe
    transcription = transcribe_audio(str(mp3_file))
    print("Transcription completed")

    # Save transcription
    with open(txt_file, 'w') as f:
        f.write(transcription)
    print(f"Transcription saved to {txt_file}")

    # Clean up MP3 file
    mp3_file.unlink()

def main(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        process_file(input_path)
    elif input_path.is_dir():
        for file in input_path.glob('*.mp4'):
            process_file(file)
    else:
        print(f"{input_path} is not a valid file or directory.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <video_file_or_directory>")
        sys.exit(1)

    main(sys.argv[1])