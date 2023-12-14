import torch


def separate_marks_for_speakers(dict_with_marks):
    speakers = {}
    for mark in dict_with_marks:
        if mark["speaker"] not in speakers:
            speakers.update({mark["speaker"]: []})

    for speaker in speakers.keys():
        for mark in dict_with_marks:
            if mark["speaker"] == speaker:
                speakers[speaker].append([mark["start"], mark["finish"]])

    return speakers


def create_separated_signals(signal, speakers_info, name, sr=16000):
    first = signal[0][
        int(speakers_info[name][0][0] * sr) : int(
            speakers_info[name][0][1] * sr
        )
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


def get_rounded_intervals(stamps):
    for speaker in stamps:
        for interval in stamps[speaker]:
            interval[0] = int(interval[0])  # math.floor
            interval[1] = int(-(-interval[1] // 1))  # math.ceil

    return stamps


def merge_intervals(stamps):
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


def sentences_with_time(timestamps, words):
    sentences = dict.fromkeys(range(len(timestamps)), "")
    _words = words[:]
    empty = []
    while _words:
        w = _words.pop(0)
        status = False
        for i in range(len(timestamps)):
            if (
                timestamps[i][0][0] <= w["start"]
                and w["start"] <= timestamps[i][0][1]
            ):
                sentences[i] += " " + w["text"]
                status = True
                break
            else:
                status = False
        if not status:
            empty.append([w["text"], w["start"]])
    for elem in sentences:
        sentences[elem] = (sentences[elem], timestamps[elem])
    return sentences, empty


def place_words(timestamps, empty):
    stamp_starts = [t[0][0] for t in timestamps]
    closest_idxs = [(99, 1)] * len(empty)
    for i in range(len(empty)):
        for j in range(len(stamp_starts)):
            if (
                min(closest_idxs[i][0], abs(stamp_starts[j] - empty[i][1]))
                != closest_idxs[i][0]
            ):
                closest_idxs[i] = (
                    min(closest_idxs[i][0], abs(stamp_starts[j] - empty[i][1])),
                    j,
                )
    return closest_idxs
