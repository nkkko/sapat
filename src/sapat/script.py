import click
from pathlib import Path


def create_transcriber(api: str, temperature: float):
    if api.lower() == "groq":
        from .transcription.groq import GroqCloudTranscription

        return GroqCloudTranscription(temperature=temperature)
    if api.lower() == "azure":
        from .transcription.azure import AzureTranscription

        return AzureTranscription(temperature=temperature)
    if api.lower() == "openai":
        from .transcription.openai import OpenAITranscription

        return OpenAITranscription(temperature=temperature)
    if api.lower() == "lemonfox":
        from .transcription.lemonfox import LemonfoxTranscription

        return LemonfoxTranscription(temperature=temperature)
    raise ValueError(f"Unsupported API: {api}")

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--language", "-l", default="en", help="Language of the audio (default: en)")
@click.option("--prompt", "-p", help="Optional prompt to guide the model")
@click.option("--temperature", "-t", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
@click.option("--quality", "-q", type=click.Choice(['L', 'M', 'H'], case_sensitive=False), default='M', help="Quality of the MP3 audio: 'L' for low, 'M' for medium, and 'H' for high (default: 'M')")
@click.option("--correct", is_flag=True, help="Use LLM to correct the transcript")
@click.option("--api", "-a", type=click.Choice(['openai', 'groq', 'azure', 'lemonfox'], case_sensitive=True), required=True, help="API to use for the transcription ('openai', 'groq', 'azure' or 'lemonfox')")
def main(input_path, language, prompt, temperature, quality, correct, api):
    """
    Transcribe video files using different APIs.

    INPUT_PATH is the path to the video file or directory containing video files.
    """
    input_path = Path(input_path)
    transcriber = create_transcriber(api, temperature)

    if input_path.is_file():
        transcriber.process_file(input_path, language, prompt, temperature, quality, correct)
    elif input_path.is_dir():
        for file in input_path.glob('*.mp4'):
            transcriber.process_file(file, language, prompt, temperature, quality, correct)
    else:
        click.echo(f"{input_path} is not a valid file or directory.")

if __name__ == "__main__":
    main()
