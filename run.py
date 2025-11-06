import argparse
from pathlib import Path
from voice_analyzer import VoiceEncoder, preprocess_wav, sampling_rate
from plotting import interactive_diarization_plot
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from time import sleep

def run_file_diarization(args):
    """Processes a single audio file and displays the diarization."""
    audio_file = Path(args.input_file)
    if not audio_file.exists():
        print(f"Error: The file '{audio_file}' was not found.")
        print("Please ensure you have downloaded the demo audio into the 'audio_data' directory.")
        return

    print(f"Processing audio file: {audio_file.name}...")
    wav = preprocess_wav(audio_file)


    segments = [[0, 5.5], [6.5, 12], [17, 25]]
    speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
    speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]

    encoder = VoiceEncoder("cpu")
    print("Running the continuous embedding on the CPU, this might take a moment...")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in
                       zip(speaker_names, speaker_embeds)}

    print("Launching interactive plot...")
    interactive_diarization_plot(similarity_dict, wav, wav_splits)

def run_live_diarization(args):
    """Runs speaker diarization in real-time using the microphone."""
    try:
        sd.query_devices()
    except Exception as e:
        print("Error: No audio devices found. Live diarization requires a microphone.")
        print(f"Details: {e}")
        return

    encoder = VoiceEncoder("cpu")
    speaker_embeds = []
    speaker_names = []

    print("--- Real-Time Speaker Diarization ---")
    
    # Record reference audio for each speaker
    while True:
        speaker_name = input(f"Enter the name for speaker {len(speaker_names) + 1} (or press Enter to start): ")
        if not speaker_name:
            if not speaker_names:
                print("Please add at least one speaker.")
                continue
            break
        speaker_names.append(speaker_name)
        
        input(f"Press Enter and speak for 5 seconds to record the reference for {speaker_name}.")
        print("Recording...")
        wav = sd.rec(int(5 * sampling_rate), samplerate=sampling_rate, channels=1)
        sd.wait()
        wav = preprocess_wav(wav.flatten())
        speaker_embeds.append(encoder.embed_utterance(wav))
        print(f"Reference for {speaker_name} recorded.")

    print("\nStarting live diarization. The plot will update in real-time. Press Ctrl+C in the terminal to stop.")

    # Live plotting setup
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=name)[0] for name in speaker_names]
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("Similarity")
    ax.set_title("Live Speaker Diarization")
    ax.legend(loc="lower right")
    fig.show()

    # Audio stream processing
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        
        try:
            wav = preprocess_wav(indata.flatten())
            if len(wav) < 1600:  # Not enough audio to process
                return

            _, cont_embeds, _ = encoder.embed_utterance(wav, return_partials=True, rate=16)
            
            # Get the latest similarity scores
            current_similarities = [cont_embeds[-1] @ embed for embed in speaker_embeds]

            # Update plot data
            for i, line in enumerate(lines):
                xdata, ydata = line.get_data()
                new_x = xdata[-1] + 1 if len(xdata) > 0 else 0
                line.set_data(np.append(xdata, new_x), np.append(ydata, current_similarities[i]))
            
            ax.relim()
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.canvas.flush_events()

        except Exception as e:
            print(f"Error during processing: {e}")


    with sd.InputStream(samplerate=sampling_rate, channels=1, callback=audio_callback):
        while True:
            sleep(0.1) # Keep the main thread alive

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""A command-line tool for speaker diarization.
        You can either run it on a pre-existing audio file or in real-time with a microphone."""
    )
    
    # Use a mutually exclusive group to ensure only one mode is selected
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_file", 
        type=str, 
        default="audio_data/X2zqiX6yL3I.mp3",
        help="Path to the audio file to process. Defaults to the demo file."
    )
    group.add_argument(
        "--live", 
        action="store_true", 
        help="Run diarization in real-time using the microphone."
    )
    
    args = parser.parse_args()

    if args.live:
        run_live_diarization(args)
    else:
        run_file_diarization(args)