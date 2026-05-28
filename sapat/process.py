# ABOUTME: File processing orchestrator for transcription pipeline
# ABOUTME: Handles convert -> split -> transcribe -> correct -> write output

import os
from pathlib import Path
from typing import Optional

import click
from halo import Halo

from sapat.audio_splitter import cleanup_chunks, should_split_file, split_audio_file
from sapat.convert import convert_audio
from sapat.providers.base import AudioFormat, TranscriptionProvider


def spinner_context(text: str):
    return Halo(text=text, spinner="dots")


def process_file(
    input_file: str,
    provider: TranscriptionProvider,
    model: str,
    language: str,
    prompt: Optional[str],
    temperature: float,
    quality: str,
    correct: bool,
):
    input_path = Path(input_file)
    fmt = provider.config.preferred_format
    ext = fmt.value
    converted_file = input_path.with_suffix(f".{ext}")
    txt_file = input_path.with_suffix(".txt")

    click.echo(click.style(f"\nProcessing: {input_file}", fg="cyan", bold=True))

    # Convert to provider-preferred format
    if not converted_file.exists():
        with spinner_context(f"Converting to {ext.upper()}...") as spinner:
            try:
                convert_audio(str(input_path), str(converted_file), quality, fmt)
                spinner.succeed(f"Conversion to {ext.upper()} completed")
            except Exception as e:
                spinner.fail(f"Conversion failed: {e}")
                return
    else:
        click.echo(click.style(f"{ext.upper()} file already exists, skipping", fg="yellow"))

    # Transcribe (with splitting if needed)
    max_size = provider.config.max_file_size_mb
    if should_split_file(str(converted_file), max_size_mb=max_size):
        click.echo(click.style(f"File is large (>{max_size}MB), splitting into chunks...", fg="yellow"))
        try:
            result = _process_large_audio(str(converted_file), provider, model, language, prompt, temperature)
        except Exception as e:
            click.echo(click.style(f"Error processing large file: {e}", fg="red"))
            return
    else:
        with spinner_context("Transcribing audio...") as spinner:
            result = provider.transcribe(
                str(converted_file),
                model=model,
                language=language,
                prompt=prompt,
                temperature=temperature,
            )
            spinner.succeed("Transcription completed")

    # Correction
    text = result.text
    if correct and provider.config.supports_correction:
        with spinner_context("Correcting transcript...") as spinner:
            text = provider.correct_transcript(text, temperature)
            spinner.succeed("Correction completed")
    elif correct and not provider.config.supports_correction:
        click.echo(
            click.style(
                f"Warning: Provider '{provider.name}' does not support transcript correction. Skipping.",
                fg="yellow",
            )
        )

    # Write output
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(text)
    click.echo(click.style(f"Transcription saved to: {txt_file}", fg="green"))

    # Cleanup
    converted_file.unlink()
    click.echo(click.style("Temporary audio file removed", fg="yellow"))


def _process_large_audio(
    audio_file: str,
    provider: TranscriptionProvider,
    model: str,
    language: str,
    prompt: Optional[str],
    temperature: float,
):
    chunk_files = []
    try:
        max_size = provider.config.max_file_size_mb
        with spinner_context("Splitting large audio file...") as spinner:
            chunk_files = split_audio_file(audio_file, max_size_mb=max_size)
            spinner.succeed(f"File split into {len(chunk_files)} chunks")

        all_texts = []
        with click.progressbar(chunk_files, label="Processing chunks") as chunks:
            for i, chunk_file in enumerate(chunks):
                try:
                    result = provider.transcribe(
                        chunk_file,
                        model=model,
                        language=language,
                        prompt=prompt,
                        temperature=temperature,
                    )
                    all_texts.append(result.text.strip())
                except Exception as e:
                    click.echo(click.style(f"Warning: chunk {i+1} failed: {e}", fg="yellow"))
                    all_texts.append(f"[Chunk {i+1} transcription failed]")

        from sapat.providers.base import TranscriptionResult
        return TranscriptionResult(text=" ".join(all_texts))

    finally:
        if chunk_files:
            cleanup_chunks(chunk_files)
            chunk_dir = os.path.dirname(chunk_files[0])
            if "sapat_chunks_" in chunk_dir:
                cleanup_chunks([chunk_dir])
