# Video Transcription Tool

This tool automates the process of transcribing video files using Azure OpenAI's Whisper API. It converts video files to MP3 format, transcribes the audio, and saves the transcription as a text file.

## Features

- Converts video files to MP3 format using ffmpeg
- Transcribes audio using Azure OpenAI's Whisper API with expanded options
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

## Creating a Dev Environment with Daytona

**Steps to Set Up Daytona Workspace**

1. Create [Daytona](https://github.com/daytonaio/daytona) Workspace:

    ```bash
    daytona create https://github.com/nkkko/sapat --code
    ```

2. There is no second step.

## Usage

Run the script with a video file or directory as an argument:

```
python script.py <video_file_or_directory> [--language <language>] [--prompt <prompt>] [--temperature <temperature>]
```

### Options

- `--language`: Specify the language of the audio (default: "en").
- `--prompt`: Optional prompt to guide the model's transcription.
- `--temperature`: The sampling temperature, between 0 and 1 (default: 0).
- `--quality`: Quality of the MP3 audio: 'L' for low, 'M' for medium, and 'H' for high (default: 'M').

Example:

```
python script.py my_video.mp4 --quality H --language es --prompt "This is a test prompt" --temperature 0.5
```

- If a file is provided, it will process that single file.
- If a directory is provided, it will process all `.mp4` files in that directory.

The script will create a `.txt` file with the same name as the input video file, containing the transcription.

## Note

This tool is designed for use with Azure OpenAI's Whisper API. Make sure you have the necessary permissions and credits to use the API.

## License

[MIT License](LICENSE)

## Claude.ai did this

```
_____     _____ ____________________     _____
  __/\____\\___ /_____\\\\\\\\\\\\\\\\\\\\\/____/___/\__
 /__/\/_____\\_//_______\\\\\\\\\\\\\\\\\\\\\/__/_/\___\/\
 \_\/    444444  666666  1111  88888   222222    \_\/  \_\/
 /\     4    4  6        1  1  8    8  2     2    /\    /\
/  \    4    4  6        1  1  8    8        2   /  \  /  \
\   \   444444  666666   1  1  888888      2    /    \/
 \  /       4   6    6   1  1  8    8    2     /  /\  /\
  \/        4   6    6   1  1  8    8   2     /  /  \/  \
  /\        4    66666  11111  888888  222222/  /        \
 /  \                                        \_/          \
/    \  >>>>>>> S.A.P.A.T. SYSTEM ONLINE <<<<<<<           \
\     \ >>>> SYNTHESIZING AUDIO PROCESSING <<<<<           /
 \    / >>>> AND TRANSCRIPTION TECHNOLOGY <<<<< __________/
  \  /  >>>>> INITIALIZING NEURAL MATRIX <<<<<</
   \/   >>>>>>>> PREPARE FOR UPLOAD <<<<<<<<<</
   /\  /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
  /  \/                                        \____________
 /      /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\
/______/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \_
```