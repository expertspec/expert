from __future__ import annotations

from os import PathLike
from typing import List, Tuple

import librosa
import torch
import torchaudio
from decord import AudioReader, bridge
from torch import Tensor

from expert.core.aggression.audio_aggression import audio_tools


bridge.set_bridge(new_bridge="torch")


class AudioAggression:
    """Extraction of aggression markers by audio channel.

    Args:
        audio (str | PathLike | Tensor): Path to the local video/audio file or Tensor object.
        stamps (List[List], optional): Specific speaker speech timestamps. Defaults to None.
        duration (int, optional): Length of intervals for extracting features. Defaults to 10.
        sr (int, optional): Sample rate. Defaults to 16000.

    Raises:
        TypeError: If innapropriate form of the record.
    """

    def __init__(
        self,
        audio: str | PathLike | Tensor,
        stamps: List[List] = None,
        duration: int = 10,
        sr: int = 16000,
    ) -> None:
        if isinstance(audio, Tensor):
            self.audio = audio
        elif isinstance(audio, str):
            self.path = audio
            self.audio = AudioReader(audio, sample_rate=sr, mono=True)
            self.audio = self.audio[:]
        else:
            raise TypeError(
                f"Type '{type(audio)}' is innapropriate for the record."
            )

        self.sr = sr
        self.frame_size = 512  # Window size of amplitude envelope.
        self.duration = duration
        self.to_voice = torchaudio.transforms.Vad(self.sr)  # For filtering.
        self.stamps = stamps
        self.loud_part = 0
        self.fast_part = 0
        self.div_aud_agg = []
        self.full_aud_agg = {}

    def _get_average(self) -> None:
        if self.stamps:
            self.fragments = dict.fromkeys(range(len(self.stamps)))
            overall_vol = 0
            overall_num = 0
            for num, stamp in enumerate(self.stamps):
                part = self.audio[0][stamp[0] * self.sr : stamp[1] * self.sr]
                part = self.to_voice(part)
                self.fragments[num] = audio_tools.amplitude_envelope(part)
                if sum(self.fragments[num]) != 0:
                    overall_vol += sum(self.fragments[num])
                    overall_num += len(self.fragments[num])
            self.average_vol = float(overall_vol / overall_num)
        else:
            self.audio = self.to_voice(self.audio)
            self.envelope = audio_tools.amplitude_envelope(self.audio[0])
            self.average_vol = float(sum(self.envelope) / len(self.envelope))

    def _get_volume(self) -> None:
        """Extraction of volume level of the fragment with fixed time window."""
        self.features = {}
        if self.stamps:
            for num, part in self.fragments.items():
                current_time = self.stamps[num][0]
                chunks = audio_tools.chunkizer(
                    chunk_length=self.duration,
                    audio=part,
                    sr=self.sr // self.frame_size,
                )
                for chunk in chunks:
                    incr, decr = audio_tools.calculate_angles(chunk)
                    marks_incr = audio_tools.get_rapidness(incr, chunk.numpy())
                    marks_decr = audio_tools.get_rapidness(decr, chunk.numpy())
                    num_changes = marks_incr + marks_decr
                    self.features.update(
                        {
                            current_time: [
                                round(float(torch.sum(chunk) / len(chunk)), 5),
                                num_changes,
                            ]
                        }
                    )
                    current_time += len(chunk) // (self.sr // self.frame_size)
        else:
            chunks = audio_tools.chunkizer(
                chunk_length=self.duration,
                audio=self.envelope,
                sr=self.sr // self.frame_size,
            )
            current_time = 0
            for chunk in chunks:
                incr, decr = audio_tools.calculate_angles(chunk)
                marks_incr = audio_tools.get_rapidness(incr, chunk.numpy())
                marks_decr = audio_tools.get_rapidness(decr, chunk.numpy())
                num_changes = marks_incr + marks_decr
                self.features.update(
                    {
                        current_time: [
                            round(float(torch.sum(chunk) / len(chunk)), 5),
                            num_changes,
                        ]
                    }
                )
                current_time += len(chunk) // (self.sr // self.frame_size)

    def _get_temp(self) -> None:
        overall_temp = 0
        overall_size = 0
        for elem in self.features:
            part = self.audio[0][
                elem * self.sr : (elem + self.duration) * self.sr
            ].numpy()
            onset_env = librosa.onset.onset_strength(y=part, sr=self.sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)[0]
            self.features[elem] = {
                "volume": self.features[elem][0],
                "dynamic_changes": self.features[elem][1],
                "temp": round(tempo, 3),
            }
            overall_temp += tempo
            overall_size += 1
        self.average_temp = overall_temp / overall_size

    def get_report(self) -> Tuple[List, List]:
        self._get_average()
        self._get_volume()
        self._get_temp()

        for elem in self.features:
            self.div_aud_agg.append(
                {
                    "time_sec": elem,
                    "volume": self.features[elem]["volume"],
                    "dynamic_changes": self.features[elem]["dynamic_changes"],
                    "temp": self.features[elem]["temp"],
                }
            )
            if self.features[elem]["volume"] > self.average_vol:
                self.loud_part += 1
            if self.features[elem]["temp"] > self.average_temp:
                self.fast_part += 1
        self.loud_part = round(self.loud_part / len(self.div_aud_agg), 2)
        self.fast_part = round(self.fast_part / len(self.div_aud_agg), 2)

        self.full_aud_agg = {
            "loud_part": self.loud_part,
            "fast_part": self.fast_part,
        }

        return (self.div_aud_agg, self.full_aud_agg)
