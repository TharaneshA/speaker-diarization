# Real-Time Speaker Diarization

This project provides a robust tool for speaker diarization, the process of partitioning an audio stream into segments according to speaker identity. It can determine "who spoke when" from both pre-recorded audio files and a live microphone stream, visualizing the results in real-time with an interactive plot.

The core of this tool is a deep learning voice encoder model that generates a high-level, fixed-dimensional representation (an embedding) of a voice. By comparing these embeddings, the system can distinguish between different speakers without needing prior training on their voices.


## Features

- **File-Based Diarization**: An# Real-Time Speaker Diarization

This project provides a robust tool for speaker diarization, the process of partitioning an audio stream into segments according to speaker identity. It can determine "who spoke when" from both pre-recorded audio files and a live microphone stream, visualizing the results in real-time with an interactive plot.

The core of this tool is a deep learning voice encoder model that generates a high-level, fixed-dimensional representation (an embedding) of a voice. By comparing these embeddings, the system can distinguish between different speakers without needing prior training on their voices.



## Features

- **File-Based Diarization**: Analyze any given audio file and automatically identify a specified number of speakers using clustering.   
- **Live Diarization**: Capture audio directly from a microphone and perform speaker diarization in real-time against pre-recorded voice references.   
- **Real-Time Visualization**: An interactive `matplotlib` plot opens for both modes, providing immediate visual feedback on speaker activity.   
- **Robust Audio Processing**: Uses a simple energy-based threshold for silence detection, making it reliable across different microphone types and environments.

---

## Project Structure

The repository is organized to be modular and easy to understand.

speaker-diarization/
├── run.py # Main executable for the project
├── plotting.py # Handles all matplotlib visualization
├── requirements.txt # Project dependencies
├── .gitignore # Specifies files to be ignored by Git
├── README.md # This documentation file
│
├── voice_analyzer/ # Core module for audio processing and voice encoding
│ ├── init.py
│ ├── audio.py
│ ├── hparams.py
│ ├── voice_encoder.py
│ └── pretrained.pt # The model weights
│
└── audio_data/ # Directory for audio files
    └── X2zqiX6yL3I.mp3 #  Example audio file

---

## Setup and Installation

Follow these steps to get the project up and running.

### 1. Clone the Repository
```sh
git clone https://github.com/TharaneshA/speaker-diarization.git  
cd speaker-diarization
```

### 2. Create a Virtual Environment (Recommended)
A virtual environment keeps your project dependencies isolated.

```sh
# For Windows
python -m venv venv
venc\Scripts\activate

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
All required Python libraries are listed in the requirements.txt file.

```sh
pip install -r requirements.txt
```

### 4. Download Pre-trained Model and Demo Audio
This project requires a pre-trained model file and an example audio file.

**Model**: Download the `pretrained.pt` file from the original Resemblyzer repository and place it inside the `voice_analyzer/` directory.

**Demo Audio**: To test the file-based mode, download the audio for the YouTube video `X2zqiX6yL3I` (using a YouTube-to-MP3 converter) and save it as `X2zqiX6yL3I.mp3` inside the `audio_data/` directory.

## Usage
The application is controlled via the `run.py` script and can be launched in two distinct modes from your terminal.

### Mode 1: Diarization from an Audio File
This mode is ideal for analyzing recordings. It processes the entire file, automatically clusters the voices to identify speakers, and displays a scrolling plot synchronized with audio playback.

**Command Syntax:**

```sh
python run.py file <path_to_audio_file> --num_speakers <number_of_speakers>
```

**Example:**

```sh
python run.py file audio_data/X2zqiX6yL3I.mp3 --num_speakers 3
```
A window will open, the audio will begin to play, and the graph will visualize who is speaking, labeling them "Speaker 1", "Speaker 2", etc.

### Mode 2: Live Diarization from Microphone
This mode uses your microphone to perform diarization in real-time. It is perfect for live demonstrations.

**Command:**

```sh
python run.py live
```

**Instructions:**

The program will first prompt you to register speakers.

Enter a name for each speaker and follow the on-screen instructions to record a 5-second voice sample.

A live volume meter will be displayed in the terminal during recording.

After registering at least one speaker, press Enter without typing a name to start the live session.

A plot window will appear and will update in real-time, always showing the name of the most similar speaker as it captures audio from your microphone.

## Troubleshooting

### Low Microphone Volume
The most common issue is the microphone input level being too low for the application to detect speech.

**Symptom:**
During reference recording, the live volume bar in the terminal barely moves. During live diarization, the plot does not update.

**Solution:**
Increase your microphone's input volume in your operating system's sound settings.

**Windows**: Go to Sound settings → Input → Device properties and increase the Volume slider.

**macOS / Linux**: Go to System Settings → Sound → Input and increase the Input volume slider.

Aim for a level where the volume meter in the application shows significant activity when you speak.alyze any given audio file and automatically identify a specified number of speakers using clustering.   
- **Live Diarization**: Capture audio directly from a microphone and perform speaker diarization in real-time against pre-recorded voice references.   
- **Real-Time Visualization**: An interactive `matplotlib` plot opens for both modes, providing immediate visual feedback on speaker activity.   
- **Robust Audio Processing**: Uses a simple energy-based threshold for silence detection, making it reliable across different microphone types and environments.

---

## Project Structure

The repository is organized to be modular and easy to understand.

speaker-diarization/
├── run.py # Main executable for the project
├── plotting.py # Handles all matplotlib visualization
├── requirements.txt # Project dependencies
├── .gitignore # Specifies files to be ignored by Git
├── README.md # This documentation file
│
├── voice_analyzer/ # Core module for audio processing and voice encoding
│ ├── init.py
│ ├── audio.py
│ ├── hparams.py
│ ├── voice_encoder.py
│ └── pretrained.pt # (Must be downloaded) The model weights
│
└── audio_data/ # Directory for audio files
    └── X2zqiX6yL3I.mp3 # (Must be downloaded) Example audio file

---

## Setup and Installation

Follow these steps to get the project up and running.

### 1. Clone the Repository
```sh
git clone https://github.com/your-username/speaker-diarization.git  
cd speaker-diarization
```

### 2. Create a Virtual Environment (Recommended)
A virtual environment keeps your project dependencies isolated.

```sh
# For Windows
python -m venv venv
venc\Scripts\activate

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
All required Python libraries are listed in the requirements.txt file.

```sh
pip install -r requirements.txt
```

### 4. Download Pre-trained Model and Demo Audio
This project requires a pre-trained model file and an example audio file.

**Model**: Download the `pretrained.pt` file from the original Resemblyzer repository and place it inside the `voice_analyzer/` directory.

**Demo Audio**: To test the file-based mode, download the audio for the YouTube video `X2zqiX6yL3I` (using a YouTube-to-MP3 converter) and save it as `X2zqiX6yL3I.mp3` inside the `audio_data/` directory.

## Usage
The application is controlled via the `run.py` script and can be launched in two distinct modes from your terminal.

### Mode 1: Diarization from an Audio File
This mode is ideal for analyzing recordings. It processes the entire file, automatically clusters the voices to identify speakers, and displays a scrolling plot synchronized with audio playback.

**Command Syntax:**

```sh
python run.py file <path_to_audio_file> --num_speakers <number_of_speakers>
```

**Example:**

```sh
python run.py file audio_data/X2zqiX6yL3I.mp3 --num_speakers 3
```
A window will open, the audio will begin to play, and the graph will visualize who is speaking, labeling them "Speaker 1", "Speaker 2", etc.

### Mode 2: Live Diarization from Microphone
This mode uses your microphone to perform diarization in real-time. It is perfect for live demonstrations.

**Command:**

```sh
python run.py live
```

**Instructions:**

The program will first prompt you to register speakers.

Enter a name for each speaker and follow the on-screen instructions to record a 5-second voice sample.

A live volume meter will be displayed in the terminal during recording.

After registering at least one speaker, press Enter without typing a name to start the live session.

A plot window will appear and will update in real-time, always showing the name of the most similar speaker as it captures audio from your microphone.

## Troubleshooting

### Low Microphone Volume
The most common issue is the microphone input level being too low for the application to detect speech.

**Symptom:**
During reference recording, the live volume bar in the terminal barely moves. During live diarization, the plot does not update.

**Solution:**
Increase your microphone's input volume in your operating system's sound settings.

**Windows**: Go to Sound settings → Input → Device properties and increase the Volume slider.

**macOS / Linux**: Go to System Settings → Sound → Input and increase the Input volume slider.

Aim for a level where the volume meter in the application shows significant activity when you speak.