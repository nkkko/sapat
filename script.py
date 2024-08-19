import os
import subprocess
import requests
from dotenv import load_dotenv
from pathlib import Path
import click
from openai import AzureOpenAI
from rev_ai import apiclient

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME_WHISPER = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME_WHISPER')
AZURE_OPENAI_API_VERSION_WHISPER = os.getenv("AZURE_OPENAI_API_VERSION_WHISPER")
AZURE_OPENAI_DEPLOYMENT_NAME_CHAT = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME_CHAT')
AZURE_OPENAI_API_VERSION_CHAT = os.getenv("AZURE_OPENAI_API_VERSION_CHAT")

# Rev AI configuration
REVAI_ACCESS_TOKEN = os.getenv('REVAI_ACCESS_TOKEN')

def convert_to_mp3(input_file, output_file, quality):
    """Convert video to MP3 using ffmpeg with specified quality"""
    if quality == 'L':
        ffmpeg_options = ['-ar', '22050', '-ac', '1', '-b:a', '96k']
    elif quality == 'M':
        ffmpeg_options = ['-ar', '44100', '-ac', '1', '-b:a', '96k']
    elif quality == 'H':
        ffmpeg_options = ['-ar', '44100', '-ac', '2', '-b:a', '192k']
    else:
        raise ValueError("Invalid quality option. Choose from 'L', 'M', 'H'.")

    command = ['ffmpeg', '-i', input_file, '-vn'] + ffmpeg_options + [output_file]
    subprocess.run(command, check=True)

def transcribe_audio_azure(
    audio_file,
    language,
    prompt,
    response_format,
    temperature,
    timestamp_granularities
):
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME_WHISPER}/audio/translations?api-version={AZURE_OPENAI_API_VERSION_WHISPER}"

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

def transcribe_audio_rev(audio_file, response_format):
    client = apiclient.RevAiAPIClient(REVAI_ACCESS_TOKEN)
    job = client.submit_job_local_file(audio_file)

    if response_format in ["json", "verbose_json"]:
        return client.get_transcript_json(job.id)
    else:
        return client.get_transcript_text(job.id)

def transcribe_audio(
    audio_file,
    provider="azure",
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
    - provider (str, optional): The API provider to use. Defaults to "azure".
    - language (str, optional): The language of the input audio. Defaults to "en".
    - prompt (str, optional): An optional text to guide the model's style or continue a previous audio segment.
    - response_format (str, optional): The format of the transcript output. Options are: "json", "text", "srt", "verbose_json", or "vtt". Defaults to "json".
    - temperature (float, optional): The sampling temperature, between 0 and 1. Defaults to 0.
    - timestamp_granularities (list, optional): List of timestamp granularities to include. Options are: "word", "segment". Defaults to None.

    Returns:
    - If response_format is "json" or "verbose_json", returns a dictionary. Otherwise, returns a string.
    """
    if provider == "azure":
        transcribe_audio_azure(audio_file, language, prompt, response_format, temperature, timestamp_granularities)
    elif provider == "rev":
        transcribe_audio_rev(audio_file, response_format)
    else:
        raise ValueError("Invalid provider. Choose from 'azure', 'rev'.")

def process_file(input_file, language, prompt, temperature, quality, correct):
    input_path = Path(input_file)
    mp3_file = input_path.with_suffix('.mp3')
    txt_file = input_path.with_suffix('.txt')

    click.echo(f"Processing {input_file}")

    if not mp3_file.exists():
        convert_to_mp3(str(input_path), str(mp3_file), quality)
        click.echo("Conversion to MP3 completed")
    else:
        click.echo("MP3 file already exists, skipping conversion")

    transcription_result = transcribe_audio(str(mp3_file), language=language, prompt=prompt, temperature=temperature)
    click.echo("Transcription completed")

    if correct:
        system_prompt = "You are a helpful assistant. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: {user provided prompt} Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."
        corrected_text = generate_corrected_transcript(0.7, system_prompt, str(mp3_file))
        click.echo("Correction completed")
    else:
        corrected_text = transcription_result

    with open(txt_file, 'w', encoding='utf-8') as f:
        if isinstance(corrected_text, dict):
            f.write(corrected_text.get('text', ''))
        else:
            f.write(corrected_text)
    click.echo(f"Transcription saved to {txt_file}")

    mp3_file.unlink()

def generate_corrected_transcript(temperature, system_prompt, audio_file):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION_CHAT
    )

    transcription = transcribe_audio(audio_file, "")
    transcription_text = transcription.get('text', '') if isinstance(transcription, dict) else transcription

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME_CHAT,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcription_text
            }
        ]
    )
    return response.choices[0].message.content


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--language", "-l", default="en", help="Language of the audio (default: en)")
@click.option("--prompt", "-p", help="Optional prompt to guide the model")
@click.option("--temperature", "-t", type=float, default=0, help="Sampling temperature (default: 0)")
@click.option("--quality", "-q", type=click.Choice(['L', 'M', 'H'], case_sensitive=False), default='M', help="Quality of the MP3 audio: 'L' for low, 'M' for medium, and 'H' for high (default: 'M')")
@click.option("--correct", is_flag=True, help="Use LLM to correct the transcript")
def main(input_path, language, prompt, temperature, quality, correct):
    """
    Transcribe video files using Azure OpenAI Whisper API.

    INPUT_PATH is the path to the video file or directory containing video files.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        process_file(input_path, language, prompt, temperature, quality, correct)
    elif input_path.is_dir():
        for file in input_path.glob('*.mp4'):
            process_file(file, language, prompt, temperature, quality, correct)
    else:
        click.echo(f"{input_path} is not a valid file or directory.")

if __name__ == "__main__":
    main()