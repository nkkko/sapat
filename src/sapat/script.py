import click
from pathlib import Path
from .transcription.groq import GroqCloudTranscription
from .transcription.azure import AzureTranscription
from .transcription.openai import OpenAITranscription
from .transcription.leopard import PicovoiceLeopardTranscription

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--language", "-l", default="en", help="Language of the audio (default: en)")
@click.option("--prompt", "-p", help="Optional prompt to guide the model")
@click.option("--temperature", "-t", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
@click.option("--quality", "-q", type=click.Choice(['L', 'M', 'H'], case_sensitive=False), default='M', help="Quality of the MP3 audio: 'L' for low, 'M' for medium, and 'H' for high (default: 'M')")
@click.option("--correct", is_flag=True, help="Use LLM to correct the transcript")
@click.option("--api", "-a", type=click.Choice(['openai', 'groq', 'azure', 'leopard'], case_sensitive=True), required=True, help="API to use for the transcription ('openai', 'groq', 'azure' or 'leopard')")
def main(input_path, language, prompt, temperature, quality, correct, api):
    """
    Transcribe video files using different APIs.

    INPUT_PATH is the path to the video file or directory containing video files.
    """
    # Initialize the correct transcription object
    input_path = Path(input_path)

    # Handle file processing based on API choice
    if api.lower() == "groq":
        transcriber = GroqCloudTranscription(temperature=temperature)
    elif api.lower() == "azure":
        transcriber = AzureTranscription(temperature=temperature)
    elif api.lower() == "openai":
        transcriber = OpenAITranscription(temperature=temperature)
    elif api.lower() == "leopard":
        transcriber = PicovoiceLeopardTranscription(temperature=temperature)
    else:
        click.echo(f"Unsupported API: {api}")
        return

    if input_path.is_file():
        transcriber.process_file(input_path, language, prompt, temperature, quality, correct)
    elif input_path.is_dir():
        for file in input_path.glob('*.mp4'):
            transcriber.process_file(file, language, prompt, temperature, quality, correct)
    else:
        click.echo(f"{input_path} is not a valid file or directory.")

if __name__ == "__main__":
    main()
