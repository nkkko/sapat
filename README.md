# Video Transcription Tool

This tool automates the process of transcribing video files using multiple transcription services: Azure OpenAI, Groq, OpenAI, and OpenRouter APIs. It converts video files to MP3 format, transcribes the audio, and saves the transcription as a text file.

## Features

- Converts video files to MP3 format using ffmpeg
- Supports transcription using Azure OpenAI, Groq, OpenAI, and OpenRouter APIs
- Supports processing of individual video files or entire directories
- Cleans up temporary MP3 files after transcription
- Provides flexibility in selecting the transcription service via configuration

## Prerequisites

- Python 3.6+
- ffmpeg installed and available in the system PATH
- API access for one or more supported services:
  - Azure OpenAI API
  - Groq Cloud API
  - OpenAI API
  - OpenRouter API

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

3. Create a `.env` file in the project root and add your API credentials:
   ```
   # Azure OpenAI
   AZURE_OPENAI_API_KEY=your_azure_api_key_here
   AZURE_OPENAI_ENDPOINT=https://DEPLOYMENTENDPOINTNAME.openai.azure.com
   AZURE_OPENAI_DEPLOYMENT_NAME_WHISPER=whisper
   AZURE_OPENAI_API_VERSION_WHISPER=2024-06-01
   AZURE_OPENAI_DEPLOYMENT_NAME_CHAT=gpt-4o
   AZURE_OPENAI_API_VERSION_CHAT=2023-03-15-preview

   # Groq
   GROQCLOUD_API_KEY=your_groq_api_key_here
   GROQCLOUD_MODEL=whisper-large-v3-turbo
   GROQCLOUD_API_ENDPOINT=https://api.groq.com/openai/v1/audio/transcriptions
   GROQCLOUD_MODEL_NAME_CHAT=llama3-8b-8192

   # OpenAI
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=whisper-1
   OPENAI_API_ENDPOINT=https://api.openai.com/v1/audio/transcriptions
   OPENAI_MODEL_NAME_CHAT=gpt-4o

   # OpenRouter
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OPENROUTER_MODEL=openai/whisper-large-v3
   OPENROUTER_API_ENDPOINT=https://openrouter.ai/api/v1/audio/transcriptions
   # Optional attribution headers for OpenRouter rankings/analytics
   OPENROUTER_HTTP_REFERER=https://github.com/nkkko/sapat
   OPENROUTER_APP_TITLE=Sapat
   ```

## Building and Installing the Package

3. **Build the Distribution:**
   Now, use build to create your package:

   ```bash
   python -m build
   ```

   This will create the wheel file in the dist directory, just like before.

4. **Install the Wheel:**
   Install as before:

   ```bash
   pip install dist/sapat-0.1.1-py3-none-any.whl  # Replace with the actual filename
   ```

5. **Add to PATH (if needed):**
   The installation process usually automatically adds the script to your PATH. If not, you'll need to add the location of the installed script to your system's PATH environment variable. The location will be something like:
   - **Linux/macOS:** `~/.local/bin`
   - **Windows:** `%USERPROFILE%\AppData\Local\Programs\Python\Python39\Scripts` (replace `Python39` with your Python version)

   For example, on Linux/macOS:

   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
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
sapat <video_file_or_directory> [--language <language>] [--prompt <prompt>] [--temperature <temperature>]
```

### Options

- `--language`: Specify the language of the audio (default: "en").
- `--prompt`: Optional prompt to guide the model's transcription.
- `--temperature`: The sampling temperature, between 0 and 1 (default: 0).
- `--quality`: Quality of the MP3 audio: 'L' for low, 'M' for medium, and 'H' for high (default: 'M').
- `--api`: Specify the API to use for transcription.
   - `--api azure` for Azure OpenAI API
   - `--api groq` for Groq Cloud API
   - `--api openai` for OpenAI API
   - `--api openrouter` for OpenRouter API

Example:

```
sapat my_video.mp4 --quality H --language es --prompt "This is a test prompt" --temperature 0.5 --api groq
```

- If a file is provided, it will process that single file.
- If a directory is provided, it will process all `.mp4` files in that directory.

The script will create a `.txt` file with the same name as the input video file, containing the transcription.

## Note

This tool is designed for use with multiple APIs (Azure OpenAI, Groq, and OpenAI). Ensure you have valid API credentials configured in the `.env` file and the necessary permissions and credits for the API service you plan to use.

## License

[MIT License](LICENSE)

## Claude.ai did the heavy lifting

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

---

This updated README includes the correct `.env` example and the additional steps for building and installing the Python package, as well as ensuring that the script is available in the PATH if needed.
