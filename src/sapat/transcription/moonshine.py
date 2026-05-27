from pathlib import Path

import click

from .base import TranscriptionBase


class MoonshineTranscription(TranscriptionBase):
    """
    Moonshine Voice implementation for local, on-device transcription.
    """

    def __init__(self, update_interval: float = 0.5):
        self.update_interval = update_interval

    def process_file(self, input_file, language, prompt, temperature, quality, correct):
        if correct:
            raise click.ClickException(
                "Moonshine does not implement transcript correction. "
                "Run without --correct or use openai, groq, or azure."
            )

        input_path = Path(input_file)
        txt_file = input_path.with_suffix('.txt')
        wav_file = input_path if input_path.suffix.lower() == '.wav' else input_path.with_suffix('.wav')
        created_wav = False

        click.echo(f"Processing {input_file}")

        if input_path.suffix.lower() != '.wav':
            if not wav_file.exists():
                self.convert_to_wav(str(input_path), str(wav_file))
                created_wav = True
                click.echo("Conversion to WAV completed")
            else:
                click.echo("WAV file already exists, skipping conversion")

        transcription_result = self.transcribe_audio(str(wav_file), language=language)
        click.echo("Transcription completed")

        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(transcription_result)
        click.echo(f"Transcription saved to {txt_file}")

        if created_wav:
            wav_file.unlink()

    def transcribe_audio(self, audio_file: str, **kwargs):
        try:
            from moonshine_voice import Transcriber, get_model_for_language, load_wav_file
        except ImportError as exc:
            raise ImportError(
                "Moonshine support requires the optional moonshine-voice package. "
                "Install it with `pip install 'sapat[moonshine]'` or "
                "`pip install moonshine-voice`."
            ) from exc

        language = kwargs.get("language") or "en"
        model_path, model_arch = get_model_for_language(language)
        audio_data, sample_rate = load_wav_file(audio_file)
        transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=self.update_interval,
        )

        try:
            transcript = transcriber.transcribe_without_streaming(audio_data, sample_rate)
            return self._transcript_to_text(transcript)
        finally:
            close = getattr(transcriber, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _transcript_to_text(transcript):
        lines = getattr(transcript, "lines", None)
        if lines is None:
            return str(transcript)

        ordered_lines = sorted(lines, key=lambda line: getattr(line, "start_time", 0.0))
        return "\n".join(
            getattr(line, "text", str(line)).strip()
            for line in ordered_lines
            if getattr(line, "text", str(line)).strip()
        )
