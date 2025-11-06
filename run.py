# run.py
import argparse
import logging
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import AgglomerativeClustering

from voice_analyzer import VoiceEncoder, preprocess_wav, sampling_rate, audio
from plotting import interactive_diarization_plot

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)-8s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def record_reference(duration=5):
    """
    Records a reference audio clip with live volume feedback in the terminal.
    """
    recorded_chunks = []
    
    def audio_callback_rec(indata, frames, time, status):
        if status:
            logger.warning(f"Recording status issue: {status}")
        
        volume_norm = np.linalg.norm(indata) * 10
        bar = '█' * int(volume_norm) + '-' * (20 - int(volume_norm))
        line_to_print = f"Recording: [{bar}]"
        print(f"\r{line_to_print:<40}", end="", flush=True)
        
        recorded_chunks.append(indata.copy())

    logger.info(f"Please speak for {duration} seconds...")
    with sd.InputStream(samplerate=sampling_rate, channels=1, callback=audio_callback_rec):
        time.sleep(duration)
    
    print()
    logger.info("Recording finished.")
    
    return np.concatenate(recorded_chunks, axis=0)


def diarize_from_file(args):
    """
    Processes a single audio file using clustering to identify speakers automatically.
    """
    input_file = Path(args.input_file)
    num_speakers = args.num_speakers

    if not input_file.exists():
        logger.error(f"The file '{input_file}' was not found.")
        return

    logger.info(f"Processing audio file: {input_file.name}")
    wav = preprocess_wav(input_file)
    encoder = VoiceEncoder("cpu")
    logger.info("Running continuous embedding on the CPU, this might take a moment...")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=8)

    if num_speakers > len(cont_embeds):
        logger.error(f"The audio is too short to find {num_speakers} speakers. Try a smaller number.")
        return

    logger.info("Clustering embeddings to identify speakers...")
    clusterer = AgglomerativeClustering(n_clusters=num_speakers).fit(cont_embeds)
    labels = clusterer.labels_

    logger.info("Generating reference embeddings for each identified speaker...")
    speaker_embeds = []
    for i in range(num_speakers):
        speaker_indices = np.where(labels == i)[0]
        speaker_embed = np.mean(cont_embeds[speaker_indices], axis=0)
        speaker_embed /= np.linalg.norm(speaker_embed)
        speaker_embeds.append(speaker_embed)

    speaker_names = [f"Speaker {i+1}" for i in range(num_speakers)]
    similarity_dict = {name: cont_embeds @ embed for name, embed in zip(speaker_names, speaker_embeds)}

    logger.info("Launching interactive plot...")
    interactive_diarization_plot(similarity_dict, wav, wav_splits)


def run_live_diarization(args):
    """
    Runs speaker diarization in real-time using the microphone with a responsive plot.
    """
    try:
        sd.query_devices()
    except Exception:
        logger.error("No audio devices found. Live diarization requires a microphone.", exc_info=True)
        return

    encoder = VoiceEncoder("cpu")
    speaker_embeds = []
    speaker_names = []

    logger.info("Real-Time Speaker Diarization.")
    
    while True:
        speaker_name = input(f"Enter name for speaker {len(speaker_names) + 1} (or press Enter to start): ")
        if not speaker_name:
            if not speaker_names:
                logger.warning("Please add at least one speaker to start.")
                continue
            break
        speaker_names.append(speaker_name)
        
        input(f"Press Enter to start recording the 5-second reference for {speaker_name}.")
        wav_ref = record_reference()
        wav = preprocess_wav(wav_ref.flatten())
        if len(wav) < sampling_rate:
            logger.error("Reference recording is too quiet or empty. Please try again and speak louder.")
            speaker_names.pop()
            continue
            
        speaker_embeds.append(encoder.embed_utterance(wav))
        logger.info(f"Reference for {speaker_name} processed successfully.")

    logger.info("Starting live diarization. Plot window is now active.")
    
    data_queue = queue.Queue()
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=name)[0] for name in speaker_names]
    
    text_label = ax.text(0.05, 0.9, "", transform=ax.transAxes, fontsize=12)
    
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("Similarity")
    ax.set_title("Live Speaker Diarization")
    ax.legend(loc="lower right")
    
    x_data, y_data = [], [[] for _ in speaker_names]
    
    def init_plot():
        for line in lines:
            line.set_data([], [])
        return lines + [text_label]

    def update_plot(frame):
        while not data_queue.empty():
            similarities = data_queue.get()
            new_x = (x_data[-1] + 1) if x_data else 0
            x_data.append(new_x)
            for i, sim in enumerate(similarities):
                y_data[i].append(sim)

            best_speaker_idx = np.argmax(similarities)
            speaker_name = speaker_names[best_speaker_idx]
            
            message = f"Speaker: {speaker_name}"
            text_label.set_text(message)
            
            _default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            color = _default_colors[best_speaker_idx % len(_default_colors)]
            text_label.set_color(color)
        
        for i, line in enumerate(lines):
            line.set_data(x_data, y_data[i])
            
        window_size = 200
        if len(x_data) > window_size:
            ax.set_xlim(x_data[-window_size], x_data[-1])
        else:
            ax.set_xlim(0, max(window_size, x_data[-1] if x_data else 1))
            
        return lines + [text_label]

    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(status)
        try:
            wav = preprocess_wav(indata.flatten())
            if len(wav) < 1600:
                return


            mels = audio.wav_to_mel_spectrogram(wav)
            mels_tensor = torch.from_numpy(mels).unsqueeze(0)
            with torch.no_grad():
                embedding = encoder(mels_tensor).cpu().numpy()[0]

            current_similarities = [embedding @ embed for embed in speaker_embeds]
            data_queue.put(current_similarities)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}", exc_info=False)

    ani = FuncAnimation(fig, update_plot, init_func=init_plot, interval=100, blit=True, cache_frame_data=False)
    
    try:
        stream = sd.InputStream(samplerate=sampling_rate, channels=1, callback=audio_callback)
        stream.start()
        logger.info("Microphone stream is active. Close the plot window to stop.")
        plt.show()

    except Exception as e:
        logger.critical(f"Failed to start audio stream or plot: {e}", exc_info=True)
    finally:
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
        logger.info("Live diarization stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A tool for speaker diarization on audio files or live microphone input.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Select the mode to run.")
    parser_file = subparsers.add_parser('file', help="Diarize an audio file.")
    parser_file.add_argument("input_file", type=str, help="Path to the audio file to process.")
    parser_file.add_argument("--num_speakers", type=int, default=2, help="Number of speakers to identify. Default: 2")
    parser_file.set_defaults(func=diarize_from_file)
    
    parser_live = subparsers.add_parser('live', help="Diarize from a live microphone stream.")
    parser_live.set_defaults(func=run_live_diarization)
    
    args = parser.parse_args()
    args.func(args)