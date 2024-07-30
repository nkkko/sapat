import os
import subprocess
import requests
from dotenv import load_dotenv
from pathlib import Path
import argparse

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

def transcribe_audio(
    audio_file,
    language="en",
    prompt=None,
    response_format="json",
    temperature=0,
    timestamp_granularities=None
):
    """
    Transcribe audio using Azure OpenAI Whisper API with expanded options.

    Parameters:
    - audio_file (str): Path to the audio file to transcribe.
    - language (str, optional): The language of the input audio. Defaults to "en".
    - prompt (str, optional): An optional text to guide the model's style or continue a previous audio segment.
    - response_format (str, optional): The format of the transcript output. Options are: "json", "text", "srt", "verbose_json", or "vtt". Defaults to "json".
    - temperature (float, optional): The sampling temperature, between 0 and 1. Defaults to 0.
    - timestamp_granularities (list, optional): List of timestamp granularities to include. Options are: "word", "segment". Defaults to None.

    Returns:
    - If response_format is "json" or "verbose_json", returns a dictionary. Otherwise, returns a string.
    """
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/audio/transcriptions?api-version=2024-02-01"

    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
    }

    data = {
        "response_format": response_format,
        "temperature": temperature
    }

    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    if timestamp_granularities:
        data["timestamp_granularities"] = timestamp_granularities

    with open(audio_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, headers=headers, data=data, files=files)

    if response.status_code == 200:
        if response_format in ["json", "verbose_json"]:
            return response.json()
        else:
            return response.text
    else:
        raise Exception(f"Transcription failed: {response.text}")

def process_file(input_file, language, prompt, temperature):
    """Process a single video file"""
    input_path = Path(input_file)
    mp3_file = input_path.with_suffix('.mp3')
    txt_file = input_path.with_suffix('.txt')

    print(f"Processing {input_file}")

    # Check if MP3 file already exists
    if not mp3_file.exists():
        # Convert to MP3
        convert_to_mp3(str(input_path), str(mp3_file))
        print("Conversion to MP3 completed")
    else:
        print("MP3 file already exists, skipping conversion")

    # Transcribe
    transcription_result = transcribe_audio(str(mp3_file), language=language, prompt=prompt, temperature=temperature)
    print("Transcription completed")

    # Save transcription
    with open(txt_file, 'w', encoding='utf-8') as f:
        if isinstance(transcription_result, dict):
            # If it's a dictionary, assume it's JSON and extract the 'text' field
            # Adjust this based on the actual structure of your JSON response
            f.write(transcription_result.get('text', ''))
        else:
            # If it's not a dictionary, assume it's already a string
            f.write(transcription_result)
    print(f"Transcription saved to {txt_file}")

    # Clean up MP3 file
    mp3_file.unlink()

def main(input_path, language, prompt, temperature):
    input_path = Path(input_path)

    if input_path.is_file():
        process_file(input_path, language, prompt, temperature)
    elif input_path.is_dir():
        for file in input_path.glob('*.mp4'):
            process_file(file, language, prompt, temperature)
    else:
        print(f"{input_path} is not a valid file or directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe video files using Azure OpenAI Whisper API.")
    parser.add_argument("input_path", help="Path to the video file or directory containing video files")
    parser.add_argument("--language", default="en", help="Language of the audio (default: en)")
    parser.add_argument("--prompt", help="Optional prompt to guide the model")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature (default: 0)")

    args = parser.parse_args()

    main(args.input_path, args.language, args.prompt, args.temperature)