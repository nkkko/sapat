# Video Transcription Tool

This tool automates the process of transcribing video files using Azure OpenAI's Whisper API. It converts video files to MP3 format, transcribes the audio, and saves the transcription as a text file.

## Features

- Converts video files to MP3 format using ffmpeg
- Transcribes audio using Azure OpenAI's Whisper API
- Supports processing of individual video files or entire directories
- Cleans up temporary MP3 files after transcription

## Prerequisites

- Python 3.6+
- ffmpeg installed and available in the system PATH
- Azure OpenAI API access

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/nkkko/sapat.git
   cd sapat
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
   ```

## Usage

Run the script with a video file or directory as an argument:

```
python script.py <video_file_or_directory>
```

- If a file is provided, it will process that single file.
- If a directory is provided, it will process all `.mp4` files in that directory.

The script will create a `.txt` file with the same name as the input video file, containing the transcription.

## Note

This tool is designed for use with Azure OpenAI's Whisper API. Make sure you have the necessary permissions and credits to use the API.

## License

[MIT License](LICENSE)