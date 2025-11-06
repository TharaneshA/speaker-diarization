# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import sleep, perf_counter as timer
from sys import stderr
from voice_analyzer.hparams import sampling_rate
try:
    import sounddevice as sd
except Exception as e:
    print(f"Could not import sounddevice, live plotting will not work: {e}")

_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def play_wav(wav, blocking=True):
    try:
        sd.play(wav, sampling_rate, blocking=blocking)
    except Exception as e:
        print(f"Failed to play audio: {e}")

def interactive_diarization_plot(similarity_dict, wav, wav_splits, x_crop=5, show_time=True):
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=name)[0] for name in similarity_dict.keys()]
    text = ax.text(0, 0, "", fontsize=10)

    def init():
        ax.set_ylim(0.4, 1)
        ax.set_ylabel("Similarity")
        if show_time:
            ax.set_xlabel("Time (seconds)")
        else:
            ax.set_xticks([])
        ax.set_title("Speaker Diarization")
        ax.legend(loc="lower right")
        return lines + [text]

    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    rate = 1 / (times[1] - times[0]) if len(times) > 1 else 1
    crop_range = int(np.round(x_crop * rate))
    ticks = np.arange(0, len(wav_splits), rate)
    ref_time = timer()

    def update(i):
        crop = (max(i - crop_range // 2, 0), i + crop_range // 2)
        ax.set_xlim(i - crop_range // 2, crop[1])
        if show_time:
            crop_ticks = ticks[(crop[0] <= ticks) & (ticks <= crop[1])]
            ax.set_xticks(crop_ticks)
            ax.set_xticklabels(np.round(crop_ticks / rate).astype(int))

        similarities = [s[i] for s in similarity_dict.values()]
        best = np.argmax(similarities)
        name, similarity = list(similarity_dict.keys())[best], similarities[best]

        if similarity > 0.75:
            message = f"Speaker: {name} (confident)"
            color = _default_colors[best % len(_default_colors)]
        elif similarity > 0.65:
            message = f"Speaker: {name} (uncertain)"
            color = _default_colors[best % len(_default_colors)]
        else:
            message = "Unknown/No speaker"
            color = "black"
        text.set_text(message)
        text.set_c(color)
        text.set_position((i, 0.96))

        for line, (name, similarities) in zip(lines, similarity_dict.items()):
            line.set_data(range(crop[0], i + 1), similarities[crop[0]:i + 1])

        current_time = timer() - ref_time
        if current_time < times[i]:
            sleep(times[i] - current_time)
        elif current_time - 0.2 > times[i]:
            import logging
            logging.warning("Animation is delayed further than 200ms!")
        return lines + [text]

    ani = FuncAnimation(fig, update, frames=len(wav_splits), init_func=init, blit=not show_time, repeat=False, interval=1)
    if wav is not None:
        play_wav(wav, blocking=False)
    plt.show()