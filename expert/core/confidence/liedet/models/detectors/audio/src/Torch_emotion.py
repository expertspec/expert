import librosa
import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset

from .AudioModel import AudioModel
from .utils import chunkizer


class Torch_emotion:
    # Заменить audio_path на False и по умолчанию передавать chunks
    def __init__(self, model, chunks=False, silence=False, audio_path=False, device="cpu", model_path=False, sr=22050):
        self.audio_path = audio_path
        self.chunks = chunks
        self.device = device
        self.silence = silence  # Категориальный признак

        # Ранее загрузка модели выполнялась здесь
        # self.model = AudioModel().to(self.device)
        # # Загрузка весов модели
        # self.model.load_state_dict(torch.load(model_path,
        #                                       map_location=device))

        self.model = model
        self.sr = sr
        self.num_samples = sr
        self.target_sample_rate = sr
        self.emos = {
            0: "anxiety",
            1: "disgust",
            2: "happiness",
            3: "boredom",
            4: "neutral",
            5: "sadness",
            6: "anger",
            7: "silence",
        }  # Словарь эмоций
        self.predict = []  # Значения эмоций

    def audio_pipeline(self):
        if self.audio_path:
            self.waveform, torch_sr = librosa.load(self.audio_path)
            # При загрузке с torchaudio иной sample rate и 2 канала
            self.chunks = chunkizer(1, self.waveform, torch_sr)
        else:
            torch_sr = self.sr
        self.chunks = [torch.Tensor(i) for i in self.chunks]  # Перевод фрагментов в тензоры
        self.test = []  # Лист для записи преобразованных фрагментов
        for i in range(len(self.chunks)):
            # Преобразование к одному значению sample rate
            w = self._resample_if_necessary(self.chunks[i], torch_sr)
            w = self._mix_down_if_necessary(w)
            w = self._cut_if_necessary(w)
            w = self._right_pad_if_necessary(w)

            mfcc = torchaudio.transforms.MFCC(sample_rate=torch_sr, n_mfcc=13)(w)
            mfcc = mfcc.repeat(3, 1, 1)  # Чтобы на вход модели подавалось 3 канала
            mfcc = np.transpose(mfcc.numpy(), (1, 2, 0))
            mfcc = np.transpose(mfcc, (2, 0, 1)).astype(np.float32)

            self.test.append(torch.tensor(mfcc, dtype=torch.float))
        # Перевод списка тензоров к тензору тензоров
        self.test = torch.stack(self.test).to(self.device)

        # Предсказание
        for i in range(self.test.shape[0]):
            c = self.test[i]
            c.unsqueeze_(0)
            # Добавление значения в список предсказаний
            self.predict.append(np.argmax(self.model(c).to("cpu").detach().numpy(), axis=1)[0])

        # Замена соответствующих значений на silence
        for num, val in enumerate(self.silence):
            if val == 1:
                self.predict[num] = 7
        self.naming()

        return self.predict

    # Перевод индексов к наименованиям
    def naming(self):
        self.predict = [self.emos[i] for i in self.predict]

    # Функции обработки аудио
    def _cut_if_necessary(self, signal):
        if signal.shape[0] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[0]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(self.sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
