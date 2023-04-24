import warnings

import librosa
import numpy as np


warnings.filterwarnings("ignore")


class ToneAnalyzer:
    def __init__(
        self,
        chunk_length,
        chunks,
        voice,
        frame_size=512,
        signal=False,
        duration=False,
        sr=22050,
    ):
        # if signal:  # Если переданы значения сигнала
        self.y = signal
        self.sr = sr
        self.duration = duration
        # else:
        #     self.load_audio(path_to_audio)
        #     self.get_duration()

        self.voice = voice
        self.chunks = chunks
        self.chunk_length = chunk_length
        self.frame_size = 512

        self.count_incr = []
        self.count_decr = []  # Количество понижений тона

        self.incr_lvl = []  # Отношение к общему числу фрагментов
        self.decr_lvl = []

        self.tone_series = []  # Временной ряд частот речевого диапазона
        self.mean_tones = []  # Основные частоты фрагментов

    def analyze_fragment(self):
        chunk_num = 0  # Переменная для записи значений временного ряда текущего фрагмента
        if self.duration > self.chunk_length:
            for i, chunk in enumerate(self.chunks):
                self.tone_series.append([])
                # Частоты речевого диапазона с заданным шагом
                self.tones = librosa.yin(
                    chunk,
                    frame_length=self.frame_size * 8,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                )
                self.tones = np.nan_to_num(self.tones)
                self.mean_tones.append(round(self.average_value(self.tones), 2))
                self.tone_series[chunk_num].append(self.tones)
                chunk_num += 1

    # Подсчет доли повышения и понижения тона
    def count_changes(self):
        for i in range(len(self.mean_tones)):
            count_decr = 0
            count_incr = 0
            for j in self.tone_series[i][0]:
                if (
                    self.mean_tones[i] + abs(np.std(self.tone_series[i][0]))
                    >= j
                    >= self.mean_tones[i] - abs(np.std(self.tone_series[i][0]))
                ):
                    if (
                        j
                        <= self.mean_tones[i]
                        - abs(np.std(self.tone_series[i][0])) / 2
                    ):  # Разделил на 2, так как иначе каждый раз получалось ноль затуханий
                        if (
                            self.mean_tones[i] - j != self.mean_tones[i]
                        ):  # Значение не равно нулю
                            count_decr += 1
                    else:
                        if self.mean_tones[i] - j != self.mean_tones[i]:
                            count_incr += 1
            self.decr_lvl.append(
                round(count_decr / len(self.tone_series[i][0]), 4)
            )
            self.incr_lvl.append(
                round(count_incr / len(self.tone_series[i][0]), 4)
            )

    def average_value(self, series):
        sum = 0
        for i in series:
            if i != 0:
                sum += i
        return sum / np.sum(series != 0)

    def load_audio(self, path_to_audio):
        self.y, self.sr = librosa.load(path_to_audio)

    def get_duration(self):
        self.duration = librosa.get_duration(self.y, self.sr)

    # Вывод параметров
    def get_results(self):
        self.results = {
            "Incr. pitch": self.incr_lvl,
            "Decr. pitch": self.decr_lvl,
            "Mean pitch": self.mean_tones,
        }
        # 'Знач-я тона' : self.tone_series}

        return self.results
