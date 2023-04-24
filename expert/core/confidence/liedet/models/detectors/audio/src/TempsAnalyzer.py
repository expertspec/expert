import warnings

import librosa


warnings.filterwarnings("ignore")


class TempsAnalyzer:
    def __init__(
        self,
        chunk_length=1,
        signal=False,
        voice=False,
        sr=22050,
        duration=False,
        chunks=False,
        path_to_audio=False,
    ):
        self.path_to_audio = path_to_audio
        self.chunks = chunks

        self.y = signal  # Аудиофрагмент
        self.duration = duration
        self.sr = sr

        self.final = {"Mean temp": []}

    def get_temp(self, fragment):
        onset_env = librosa.onset.onset_strength(y=fragment, sr=self.sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)
        return round(tempo[0], 1)

    def get_analyze(self):
        for chunk in self.chunks:
            self.final["Mean temp"].append(self.get_temp(chunk))
        return self.final
