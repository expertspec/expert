import librosa
from heapq import merge
from datetime import datetime


class DialogStats:
    def __init__(self, audio, sr, annotation, speakers):
        self.audio = audio
        self.sr = sr
        self.annotation = annotation
        self.speakers = speakers

    # Извлечение интервалов
    def get_timestamps(self):
        self.timestamps = []
        for elem in self.annotation:
            self.timestamps.append(self.annotation[elem][1][0])

    # Когда была речь, в целом
    def non_overlapping_intervals(self):
        self.non_overlapping = []
        starts = sorted(
            [(i[0], 1) for i in self.timestamps]
        )  # start of interval adds a layer of overlap
        ends = sorted([(i[1], -1) for i in self.timestamps])  # end removes one
        layers = 0
        current = []
        for value, event in merge(
            starts, ends
        ):  # sorted by value, then ends (-1) before starts (1)
            layers += event
            if layers == 1:  # start of a new non-overlapping interval
                current.append(value)
            elif (
                current
            ):  # we either got out of an interval, or started an overlap
                current.append(value)
                self.non_overlapping.append(current)
                current = []
        return self.non_overlapping

    # Перебивания
    def get_intersections(self):
        def _get_intersection(interval1, interval2):
            new_min = max(interval1[0], interval2[0])
            new_max = min(interval1[1], interval2[1])
            return [new_min, new_max] if new_min <= new_max else None

        self.interval_intersection = [
            x
            for x in (
                _get_intersection(y, z)
                for y in self.speakers["SPEAKER_01"]
                for z in self.speakers["SPEAKER_00"]
            )
            if x is not None
        ]

    # Продолжительность разговора
    def call_duration(self):
        if len(self.audio.shape) == 2:
            self.duration = self.audio.shape[1] / self.sr
        elif len(self.audio.shape) == 1:
            self.duration = self.audio.shape[0] / self.sr
        else:
            raise TypeError

    # Продолжительность тишины в начале/конце разговора
    def start_end_silence(self):
        """
        Example of annotation's element {0: (' поступила заявка от вашего личного кабинета',
        ([1.257, 4.497], 'SPEAKER_01'))}
        """
        silence_in_the_start = self.annotation[0][1][0][0]
        self.last_key = list(self.annotation)[
            -1
        ]  # index of the last annotation record
        silence_in_the_end = (
            self.duration - self.annotation[self.last_key][1][0][1]
        )
        return silence_in_the_start, silence_in_the_end

    # Длительность речи. Считается как общая (не пересекающиеся интервалы), так и для каждого спикера/одновременной речи
    def get_voice_duration(self, speaker=None, intersactions=None):
        if speaker:
            intervals = self.speakers[speaker]
        elif intersactions:
            intervals = self.interval_intersection
        else:
            intervals = self.non_overlapping
        voice_duration = 0
        for interval in intervals:
            voice_duration += interval[1] - interval[0]

        return voice_duration

    # Максимальное значение тишины
    def silence_info(self):
        silences = []
        for i in range(1, len(self.non_overlapping)):
            silences.append(
                self.non_overlapping[i][0] - self.non_overlapping[i - 1][1]
            )

        silence_part = (
            self.duration - self.get_voice_duration()
        ) / self.duration
        return max(silences), silence_part

    # Средняя скорость речи выбранного спикера
    def get_tempo(self, speaker):
        each_temp = []
        if len(self.audio.shape) == 2:
            audio = self.audio[0]
        else:
            audio = self.audio
        for interval in self.speakers[speaker]:
            fragment = audio[
                int(self.sr * interval[0]) : int(self.sr * interval[1])
            ].numpy()
            onset_env = librosa.onset.onset_strength(fragment, sr=self.sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)
            each_temp.append(tempo)
        return float((sum(each_temp) / len(each_temp)))

    def interrupt_info(self):
        speaker_0_interrupt = 0
        speaker_1_interrupt = 0
        for interval in self.interval_intersection:
            for elem in self.speakers["SPEAKER_00"]:
                # Перебивает спикер 1
                if interval[0] <= elem[1] and interval[0] > elem[0]:
                    print("Перебивал SPEAKER_01")
                    print(
                        datetime.fromtimestamp(interval[0]).strftime("%M:%S")
                        + " - "
                        + datetime.fromtimestamp(interval[1]).strftime("%M:%S")
                    )
                    speaker_1_interrupt += 1
            for elem in self.speakers["SPEAKER_01"]:
                # Перебивает спикер 0
                if interval[0] <= elem[1] and interval[0] > elem[0]:
                    print("Перебивал SPEAKER_00")
                    print(
                        datetime.fromtimestamp(interval[0]).strftime("%M:%S")
                        + " - "
                        + datetime.fromtimestamp(interval[1]).strftime("%M:%S")
                    )
                    speaker_0_interrupt += 1
        return speaker_0_interrupt, speaker_1_interrupt


def call_statistic(audio, sr, annotation, speakers):
    report = {}
    dialStats = DialogStats(audio, sr, annotation, speakers)
    dialStats.get_timestamps()
    dialStats.call_duration()
    dialStats.get_intersections()
    dialStats.non_overlapping_intervals()
    report.update({"Продолжительность разговора сек": [dialStats.duration]})
    start_silence, end_silence = dialStats.start_end_silence()
    report.update({"Тишина в начале сек": [start_silence]})
    report.update({"Тишина в конце сек": [end_silence]})
    report.update(
        {
            "Длителньость тишины сек": [
                dialStats.duration - dialStats.get_voice_duration()
            ]
        }
    )
    report.update(
        {
            "Доля тишины": [
                dialStats.duration
                / (dialStats.duration - dialStats.get_voice_duration())
            ]
        }
    )
    report.update(
        {
            "Продолжительность речи SPEAKER_00": [
                dialStats.get_voice_duration(speaker="SPEAKER_00")
            ]
        }
    )
    report.update(
        {"Скорость речи SPEAKER_00": [dialStats.get_tempo("SPEAKER_00")]}
    )
    report.update(
        {"Количество интервалов SPEAKER_00": [len(speakers["SPEAKER_00"])]}
    )
    report.update(
        {
            "Доля речи SPEAKER_00": [
                dialStats.get_voice_duration("SPEAKER_00")
                / dialStats.get_voice_duration()
            ]
        }
    )
    report.update(
        {
            "Продолжительность речи SPEAKER_01": [
                dialStats.get_voice_duration(speaker="SPEAKER_01")
            ]
        }
    )
    report.update(
        {"Скорость речи SPEAKER_01": [dialStats.get_tempo("SPEAKER_01")]}
    )
    report.update(
        {"Количество интервалов SPEAKER_01": [len(speakers["SPEAKER_01"])]}
    )
    report.update(
        {
            "Доля речи SPEAKER_01": [
                dialStats.get_voice_duration("SPEAKER_01")
                / dialStats.get_voice_duration()
            ]
        }
    )
    report.update(
        {
            "Продолжительность одновременной речи": [
                dialStats.get_voice_duration(intersactions=True)
            ]
        }
    )
    report.update(
        {
            "Доля одновременной речи": [
                dialStats.get_voice_duration(intersactions=True)
                / dialStats.duration
            ]
        }
    )
    speaker_0_interrupt, speaker_1_interrupt = dialStats.interrupt_info()
    report.update({"Количество перебиваний SPEAKER_00": [speaker_1_interrupt]})
    report.update({"Количество перебиваний SPEAKER_01": [speaker_0_interrupt]})

    return report
