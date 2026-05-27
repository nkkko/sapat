from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import click


class TranscriptionBase(ABC):
    """
    Base class for transcription APIs.
    """

    @staticmethod
    def convert_to_mp3(input_file: str, output_file: str, quality: str):
        """
        Converts an audio file to MP3 format using FFmpeg.

        Parameters:
        - input_file (str): Path to the input file.
        - output_file (str): Path to the output MP3 file.
        - quality (str): Desired audio quality ('L', 'M', 'H').
        """
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

    @staticmethod
    def convert_to_wav(input_file: str, output_file: str):
        """
        Converts an audio or video file to 16 kHz mono WAV for local ASR engines.
        """
        command = [
            'ffmpeg',
            '-i',
            input_file,
            '-vn',
            '-ar',
            '16000',
            '-ac',
            '1',
            '-sample_fmt',
            's16',
            output_file,
        ]
        subprocess.run(command, check=True)


    def process_file(self, input_file, language, prompt, temperature, quality, correct):
        input_path = Path(input_file)
        mp3_file = input_path.with_suffix('.mp3')
        txt_file = input_path.with_suffix('.txt')

        click.echo(f"Processing {input_file}")

        if not mp3_file.exists():
            self.convert_to_mp3(str(input_path), str(mp3_file), quality)
            click.echo("Conversion to MP3 completed")
        else:
            click.echo("MP3 file already exists, skipping conversion")

        transcription_result = self.transcribe_audio(str(mp3_file), language=language, prompt=prompt, temperature=temperature)
        click.echo("Transcription completed")

        if correct:
            system_prompt = "You are a helpful assistant. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: {user provided prompt} Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided."
            corrected_text = self.generate_corrected_transcript(str(mp3_file), 0.7, system_prompt)
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

    @abstractmethod
    def transcribe_audio(self, audio_file: str, **kwargs):
        """
        Abstract method for transcribing audio files. Must be implemented by subclasses.

        Parameters:
        - audio_file (str): Path to the audio file to be transcribed.
        - kwargs: Additional arguments for transcription.

        Returns:
        - The transcription result (implementation-specific).
        """
        pass

    @staticmethod
    def generate_corrected_transcript(audio_file, temperature, prompt):
        """
        Default implementation for generating corrected transcripts. Can be overridden by subclasses.
        """
        raise NotImplementedError("Correction not implemented for this API.")
      
