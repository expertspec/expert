import json
import math
import os
import re
import warnings

import librosa
import noisereduce as nr
import numpy as np
from decord import AudioReader, bridge
from pydub import AudioSegment

bridge.set_bridge(new_bridge="torch")

warnings.filterwarnings("ignore")


def test_mfcc(
    dataset_path,
    json_path,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    num_segments=2,
    sr=22050,
    duration=2,
):
    """
    Функция позволяет сохранять mfcc каждого аудиотрека в указанной директории в json файл
    """
    SAMPLE_RATE = sr
    DURATION = duration
    SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION
    data = {"mapping": [], "name": [], "mfcc": []}

    num_samples_per_segment = int(SAMPLE_PER_TRACK / num_segments)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Обработка файлов
        for f in filenames:
            # Загрузка аудиофайлов
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Извлечение mfcc и сбор данных

            for s in range(num_segments):
                start_sample = num_samples_per_segment * s
                finish_sample = start_sample + num_samples_per_segment

                if len(signal) < finish_sample:
                    mfcc = librosa.feature.mfcc(
                        signal[start_sample:],
                        sr=sr,
                        n_fft=n_fft,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length,
                    )
                else:
                    mfcc = librosa.feature.mfcc(
                        signal[start_sample:finish_sample],
                        sr=sr,
                        n_fft=n_fft,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length,
                    )
                mfcc = mfcc.T

                # Запись значений
                data["name"].append(f)
                data["mfcc"].append(mfcc.tolist())
                print("{}, segment:{}".format(file_path, s + 1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def load(dataset_path):
    """

    Загрузка и преобразование данных из JSON
    """
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    # lists to numpy array
    inputs = np.array(data["mfcc"])
    names = np.array(data["name"])
    inputs = inputs[..., np.newaxis]

    return inputs, names


# Делит аудио файл(фрагменты) на чанки
def chunkizer(chunk_length, audio, sr):
    num_chunks = math.ceil(librosa.get_duration(audio, sr=sr) / chunk_length)
    chunks = []
    for i in range(num_chunks):
        chunks.append(
            audio[i * chunk_length * sr : (i + 1) * chunk_length * sr]
        )
    return chunks


def prepare_audio_from_video(video_path):
    audio = AudioReader(video_path, sample_rate=22050, mono=True)
    sr = 22050
    audio = audio[:]
    video_name = os.path.basename(video_path).split(".")[0]
    folder = os.path.dirname(video_path)
    audio_path = os.path.join(folder, video_name + ".mp3")
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    # перевод librosa в pydub
    audio = np.array(reduced_noise * (1 << 15), dtype=np.int16)
    audio = AudioSegment(
        audio.tobytes(),
        frame_rate=sr,
        sample_width=audio.dtype.itemsize,
        channels=1,
    )
    normalized = normalization(audio, -18.0)
    normalized.export(audio_path, format="mp3")

    return audio_path


# Выравнивает громкость к заданному среднему значению
def normalization(signal, target_dBFS):
    change_in_dBFS = target_dBFS - signal.dBFS
    return signal.apply_gain(change_in_dBFS)
