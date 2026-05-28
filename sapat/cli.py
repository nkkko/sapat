# ABOUTME: Click CLI for the sapat transcription tool
# ABOUTME: Dynamically discovers available providers from the registry

from pathlib import Path

import click

from sapat.__version__ import __version__
from sapat.process import process_file
from sapat.providers import get_available_providers, get_provider

VERSION = __version__


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--provider", "-p",
    default=None,
    help="Service provider for STT (choices built dynamically from available providers)",
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Model to use for STT (provider-specific)",
)
@click.option("--language", "-l", default="en", help="Language of the audio (default: en)")
@click.option("--transcription-prompt", "-t", help="Optional prompt to guide the model")
@click.option(
    "--temperature", "-temp",
    type=float,
    default=0,
    help="Sampling temperature (default: 0)",
)
@click.option(
    "--quality", "-q",
    type=click.Choice(["L", "M", "H"], case_sensitive=False),
    default="L",
    help="Audio quality: L=low, M=medium, H=high (default: L)",
)
@click.option("--correct", is_flag=True, help="Correct transcript with LLM")
@click.version_option(version=VERSION)
def main(input_path, provider, model, language, transcription_prompt, temperature, quality, correct):
    """
    Transcribe video files using the specified STT service.

    INPUT_PATH is the path to the video file or directory containing video files.
    """
    available = get_available_providers()

    if not available:
        raise click.ClickException(
            "No providers available. Set API keys in .env or install optional dependencies."
        )

    if provider is None:
        provider = "azure" if "azure" in available else sorted(available.keys())[0]

    if provider not in available:
        raise click.ClickException(
            f"Provider '{provider}' is not available. "
            f"Available: {', '.join(sorted(available.keys()))}"
        )

    prov_cls = available[provider]
    prov_instance = prov_cls()

    if model is None:
        model = prov_cls.config.default_model
    model = prov_instance.resolve_model(model)

    click.echo(click.style(f"SAPAT v{VERSION}", fg="blue", bold=True))
    click.echo(click.style("Speech-to-Text Transcription Tool", fg="blue"))
    click.echo("-----------------------------------")
    click.echo(click.style(f"Provider: {provider}", fg="cyan"))
    click.echo(click.style(f"Model: {model}", fg="cyan"))
    click.echo(click.style(f"Language: {language}", fg="cyan"))
    click.echo("-----------------------------------")

    input_path = Path(input_path)

    if input_path.is_file():
        process_file(
            str(input_path), prov_instance, model, language,
            transcription_prompt, temperature, quality, correct,
        )
    elif input_path.is_dir():
        files = list(input_path.glob("*.mp4"))
        with click.progressbar(files, label="Processing files") as bar:
            for file in bar:
                process_file(
                    str(file), prov_instance, model, language,
                    transcription_prompt, temperature, quality, correct,
                )
    else:
        click.echo(click.style(f"Error: {input_path} is not valid.", fg="red"))

    click.echo(click.style("\nProcessing complete!", fg="green", bold=True))


if __name__ == "__main__":
    main()
