import warnings
from math import ceil, isnan

import librosa
import numpy as np

warnings.filterwarnings("ignore")


class VolumeAnalyzer:
    def __init__(
        self,
        chunk_length,
        normalization=True,
        frame_size=512,
        signal=None,
        sr=22050,
        voice=None,
        path_to_audio=False,
    ):
        if isinstance(signal, np.ndarray):  # Если переданы значения сигнала
            self.y = signal[0]
            self.sr = sr
            self.get_duration()
        else:
            self.load_audio(path_to_audio)
            self.get_duration()
        if voice:  # Если переданы параметры частот относящихся к речевым
            self.voice = voice
        else:
            self.voice = librosa.yin(
                self.y,
                frame_length=2048,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("A5"),
            )  # Выделение голоса
            for i in range(len(self.voice)):
                if self.voice[i] > librosa.note_to_hz("A5") or self.voice[
                    i
                ] < librosa.note_to_hz("C2"):
                    self.voice[i] = np.nan
        self.chunk_length = chunk_length
        self.chunks = []
        self.frame_size = frame_size
        self.normalization = normalization

        self.count_incr = []
        self.count_decr = []  # Количество понижений громкости

        self.incr_lvl = []  # Отношение к общему числу фрагментов
        self.decr_lvl = []

        self.voice_vols = []
        self.average_vols = []

        self.noise_lvl = (
            []
        )  # Доля отброшенных фрагментов (не относящихся к голосовому диапазону)
        self.ampl_series = []
        self.silence = (
            []
        )  # Категориальный признак, указывающий на то, является ли фрагмент тишиной

    # Делит аудио файл(фрагменты) на чанки
    def chunkizer(self, audio, sr):
        num_chunks = ceil(
            librosa.get_duration(y=audio, sr=sr) / self.chunk_length
        )
        for i in range(num_chunks):
            self.chunks.append(
                audio[
                    i
                    * self.chunk_length
                    * sr : (i + 1)
                    * self.chunk_length
                    * sr
                ]
            )
        return self.chunks

    # Определение амплитудных значений фрагмента
    def analyze_fragment(self):
        if self.duration > self.chunk_length:
            chunks_amplitudes = []
            self.chunks = self.chunkizer(self.y, self.sr)
            for chunk in self.chunks:
                amplitude_envelope = []
                for i in range(0, len(chunk), self.frame_size):
                    current = max(chunk[i : i + self.frame_size])
                    amplitude_envelope.append(
                        current
                    )  # Запись амплитудных значений
                chunks_amplitudes.append(amplitude_envelope)
            self.voice_vols = chunks_amplitudes

        else:
            amplitude_envelope = []
            # Используется фрагмент целиком
            for i in range(0, len(self.y), self.frame_size):
                current = max(self.y[i : i + self.frame_size])
                amplitude_envelope.append(
                    current
                )  # Запись амплитудных значений

            self.voice_vols.append(amplitude_envelope)

        # Удаление амплитудных значений не голосового диапазона
        for i in range(len(self.voice_vols)):
            new_fragments_param = (
                []
            )  # Заменим значения на значения актуальные только для голоса
            ampl_len = len(
                self.voice_vols[i]
            )  # Кол-во амплитудных значений фрагмента
            for ampl in range(len(self.voice_vols[i]) - 1):
                try:
                    if not isnan(self.voice[i * ampl_len + ampl]):
                        new_fragments_param.append(self.voice_vols[0][i][ampl])
                # При делении на фрагменты длительностью менее 10 секунд возникают ошибки
                except IndexError:
                    new_fragments_param.append(self.voice_vols[i][ampl])

            self.noise_lvl.append(
                round(
                    np.nansum(self.voice_vols[i]) / len(self.voice_vols[i]), 5
                )
            )
            self.voice_vols[i] = new_fragments_param
            # Так как возникают пустые фрагменты
            if len(new_fragments_param) == 0:
                self.voice_vols[i] = [0]

    # Подсчет доли повышения громкости и понижения
    def count_changes(self):
        self.average_vols = [round(np.mean(i), 5) for i in self.voice_vols]
        for i in range(len(self.voice_vols)):
            count_decr = 0
            count_incr = 0
            for j in self.voice_vols[i]:
                if (
                    self.average_vols[i] + abs(np.std(self.voice_vols[i]))
                    >= j
                    >= self.average_vols[i] - abs(np.std(self.voice_vols[i]))
                ):
                    if (
                        j
                        <= self.average_vols[i]
                        - abs(np.std(self.voice_vols[i])) / 2
                    ):  # Разделил на 2, так как
                        # иначе каждый раз получалось ноль затуханий
                        count_decr += 1
                    else:
                        count_incr += 1
            self.decr_lvl.append(round(count_decr / len(self.voice_vols[i]), 4))
            self.incr_lvl.append(round(count_incr / len(self.voice_vols[i]), 4))
        self.is_silence()

    def load_audio(self, path_to_audio):
        self.y, self.sr = librosa.load(path_to_audio)

    def get_duration(self):
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)

    # Получение амплитундой огибающей с заданным шагом
    def ampl_time(self):
        chunk_num = 0
        if self.duration > self.chunk_length:
            for chunk in self.chunks:
                self.ampl_series.append([])
                for i in range(0, len(chunk), self.frame_size):
                    current = max(chunk[i : i + self.frame_size])
                    self.ampl_series[chunk_num].append(current)
                chunk_num += 1

    # Функция для определения тишины
    def is_silence(self):
        # Подобрано эмпирически для нормализованной записи
        if self.normalization:
            SILENCE_LEVEL = librosa.power_to_db(np.mean(abs(self.y))) - 6
        else:
            SILENCE_LEVEL = librosa.power_to_db(np.max(abs(self.y))) - 11
        for vol in self.average_vols:
            if librosa.power_to_db(vol) < SILENCE_LEVEL:
                self.silence.append(1)
            else:
                self.silence.append(0)

    # Вывод параметров
    def get_results(self):
        self.results = {
            "Increase vol": np.nan_to_num(self.incr_lvl),
            "Decrease vol": np.nan_to_num(self.decr_lvl),
            "Mean vol": np.nan_to_num(self.average_vols),
            "Noise part": np.nan_to_num(self.noise_lvl),
        }
        # 'Огибающие': self.ampl_series}

        return self.results
