import torch
import numpy as np


def amplitude_envelope(signal, frame_size=512):
    amplitude_envelope = []
    for frame in range(0, len(signal), frame_size):
        current = max(signal[frame : frame + frame_size].numpy())
        amplitude_envelope.append(current)
    return torch.tensor(amplitude_envelope)


def chunkizer(chunk_length, audio, sr):
    duration = audio.shape[0] / sr
    num_chunks = int(-(-duration // chunk_length))  # ceil without import math
    chunks = []
    for i in range(num_chunks):
        chunks.append(audio[i * chunk_length * sr:(i + 1) * chunk_length * sr])
    return chunks


def calculate_angles(envelope):
    # Returns 2 dictionaries with angles and timestams for increasing and decreasing
    increases = {}
    decreases = {}
    for i in range(len(envelope) - 1):
        angle = np.rad2deg(np.arctan(envelope[i + 1] - envelope[i]))
        if angle >= 0:
            increases.update({i + 1: float(angle)})
        else:
            decreases.update({i + 1: float(angle)})
    # To calculate the last angle
    if len(envelope) != len(increases) + len(decreases):
        try:
            angle = np.rad2deg(np.arctan(envelope[-1] - envelope[-2]))
        except IndexError:
            angle = 0
        if angle >= 0:
            increases.update({len(envelope) - 1: float(angle)})
        else:
            decreases.update({len(envelope) - 1: float(angle)})
    return increases, decreases


def get_rapidness(sequence, envelope):
    sharp_angles = {}
    for timestamp, angle in sequence.items():
        if angle > 0:
            if angle >= np.mean(
                list(sequence.values())) + 2 * np.std(list(sequence.values())
                ):
                sharp_angles.update({timestamp: angle})
        else:
            if angle <= np.mean(
                list(sequence.values())) - 2 * np.std(list(sequence.values())
                ):
                sharp_angles.update({timestamp: angle})

    timestamps = []
    # Loudness detection
    for timestamp in sharp_angles.keys():
        try:
            if 1 - np.abs(
                envelope[timestamp]) < np.abs(envelope[timestamp]) - np.abs(np.mean(envelope)
                ):
                timestamps.append(timestamp)
        except IndexError:
            continue
    return len(timestamps)

