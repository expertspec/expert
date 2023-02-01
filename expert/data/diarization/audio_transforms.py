from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Dict
from os import PathLike


def rttm_to_timestamps(path: str | PathLike) -> List:
    with open(path, "r", encoding="utf-8") as file:
        lines = []
        for line in file:
            lines.append(line)
    
    timestamps = []
    for line in lines:
        line = line.split(" ")
        timestamps.append(
            {
                "speaker": line[-3],
                "start": float(line[3]),
                "finish": round(float(line[3]) + float(line[4]), 3),
            }
        )
    
    return timestamps


def json_to_timestamps(annot: Dict) -> List:
    timestamps = []
    for line in annot["content"]:
        timestamps.append(
            {
                "speaker": line["label"],
                "start": round(line["segment"]["start"], 3),
                "finish": round(line["segment"]["end"], 3),
            }
        )
    
    return timestamps


def separate_marks_for_speakers(dict_with_marks: List) -> Dict:
    speakers = {}
    for mark in dict_with_marks:
        if mark["speaker"] not in speakers:
            speakers.update({mark["speaker"]: []})

    for speaker in speakers.keys():
        for mark in dict_with_marks:
            if mark["speaker"] == speaker:
                speakers[speaker].append([mark["start"], mark["finish"]])
    
    return speakers


def create_separated_signals(signal: List, speakers_info: Dict, name: str, sr: int = 16000) -> Tensor:
    first = signal[0][
        int(speakers_info[name][0][0] * sr) : int(speakers_info[name][0][1] * sr)
    ]
    for num in range(1, len(speakers_info[name])):
        first = torch.concat(
            (
                first,
                signal[0][
                    int(speakers_info[name][num][0] * sr) : int(
                        speakers_info[name][num][1] * sr
                    )
                ],
            )
        )
    
    return first


def get_rounded_intervals(stamps: Dict) -> Dict:
    for speaker in stamps:
        for interval in stamps[speaker]:
            interval[0] = int(interval[0]) # math.floor
            interval[1] = int(-(-interval[1] // 1)) # math.ceil
    
    return stamps


def merge_intervals(stamps: Dict) -> Dict:
    for speaker in stamps:
        intervals = stamps[speaker]
        # Merge overlapped intervals.
        stack = []
        # Insert first interval into stack.
        stack.append(intervals[0])
        for i in intervals[1:]:
            # Check for overlapping interval.
            if stack[-1][0] <= i[0] <= stack[-1][-1]:
                stack[-1][-1] = max(stack[-1][-1], i[-1])
            else:
                stack.append(i)
        stamps[speaker] = stack
    
    return stamps