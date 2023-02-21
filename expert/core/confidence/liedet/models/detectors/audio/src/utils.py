import librosa
import numpy as np
import os
import re

import math
import json
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, effects
from moviepy.editor import VideoFileClip

import warnings

warnings.filterwarnings('ignore')


def test_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=2,
              sr=22050, duration=2):
    '''
    Функция позволяет сохранять mfcc каждого аудиотрека в указанной директории в json файл
    '''
    SAMPLE_RATE = sr
    DURATION = duration
    SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION
    data = {
        "mapping": [],
        "name": [],
        "mfcc": []
    }

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
                    mfcc = librosa.feature.mfcc(signal[start_sample:],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                else:
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                mfcc = mfcc.T

                # Запись значений
                data['name'].append(f)
                data['mfcc'].append(mfcc.tolist())
                print("{}, segment:{}".format(file_path, s + 1))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def load(dataset_path):
    '''

    Загрузка и преобразование данных из JSON
    '''
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    # lists to numpy array
    inputs = np.array(data['mfcc'])
    names = np.array(data['name'])
    inputs = inputs[..., np.newaxis]

    return inputs, names


# Делит аудио файл(фрагменты) на чанки
def chunkizer(chunk_length, audio, sr):
    num_chunks = math.ceil(librosa.get_duration(audio, sr=sr) / chunk_length)
    chunks = []
    for i in range(num_chunks):
        chunks.append(audio[i * chunk_length * sr:(i + 1) * chunk_length * sr])
    return chunks

def prepare_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    # Извлечение имени файла из пути, для создания одноименного аудиофайла
    template = r"\w+.mp4$"
    video_name = re.search(template, video_path)
    folder = video_path.replace(video_name[0], '')
    video_name = video_name[0][:-4]
    audio_path = os.path.join(folder, video_name + '.mp3')
    video.audio.write_audiofile(audio_path) # Сохранение аудиофайла

    # Предобработка звука
    audio, sr = librosa.load(audio_path)
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    # перевод librosa в pydub
    audio = np.array(reduced_noise* (1<<15), dtype=np.int16)
    audio = AudioSegment(
        audio.tobytes(),
        frame_rate=sr,
        sample_width=audio.dtype.itemsize,
        channels=1
    )
    normalized = normalization(audio, -18.0)
    normalized.export(audio_path, format='mp3')

    return audio_path

# Эффекты
def stretch(data, rate=0.75):
    return librosa.effects.time_stretch(data, rate)


def pitch(data, sampling_rate, n_steps):
    return librosa.effects.pitch_shift(data, sampling_rate, n_steps=n_steps, bins_per_octave=12)

# Создает дополнительные экземляры аудио с небольшими изменеиями
def augmentation(folder_name, audio_name):
    audio, sr = librosa.load(folder_name + '/' + audio_name)
    pitch_up = pitch(audio, sr, 3)
    pitch_down = pitch(audio, sr, -2)
    stretched = stretch(audio)

    sf.write(folder_name + '/' + 'up_' + audio_name[:-4] + '.wav', pitch_up, sr)
    sf.write(folder_name + '/' + 'down_' + audio_name[:-4] + '.wav', pitch_down, sr)
    sf.write(folder_name + '/' + 'str_' + audio_name[:-4] + '.wav', stretched, sr)

# Выравнивает громкость к заданному среднему значению
def normalization(signal, target_dBFS):
    change_in_dBFS = target_dBFS - signal.dBFS
    return signal.apply_gain(change_in_dBFS)
