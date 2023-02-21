import os
import sys
from pathlib import Path

import pandas as pd
from Signal_Analysis.features import signal as sig
from sklearn.preprocessing import OneHotEncoder

import torch
from torch import load

from .src.AudioModel import AudioModel
from .src.TempsAnalyzer import TempsAnalyzer
from .src.ToneAnalyzer import ToneAnalyzer
from .src.Torch_emotion import Torch_emotion
from .src.utils import normalization, prepare_audio_from_video
from .src.VolumeAnalyzer import VolumeAnalyzer


def features_to_timeseries(
    model_path, device, chunk_length=1, signal=None, video_path=None, sr=22050, audio_path=None, normalization=True
):
    results = {}
    if video_path:
        audio_path = prepare_audio_from_video(video_path)
    # Выделение параметров громкости
    volume = VolumeAnalyzer(
        normalization=normalization, chunk_length=chunk_length, sr=sr, signal=signal, path_to_audio=audio_path
    )
    signal, sr, silence, duration, chunks, voice = (
        volume.y,
        volume.sr,
        volume.silence,
        volume.duration,
        volume.chunks,
        volume.voice,
    )

    volume.analyze_fragment()
    volume.count_changes()
    results.update(volume.get_results())
    # Выделение параметров темпа
    temp = TempsAnalyzer(signal=signal, sr=sr, voice=voice, duration=duration, chunks=chunks)
    results.update(temp.get_analyze())
    # Выделение параметров тона
    tone = ToneAnalyzer(chunk_length=chunk_length, chunks=chunks, voice=voice, signal=signal, duration=duration)
    tone.analyze_fragment()
    tone.count_changes()
    results.update(tone.get_results())
    # Выделение дрожи в голосе
    jitters = {"jitter_local": [], "jitter_rap": []}
    # Обработка происходит последовательно для каждого фрагмента
    for chunk in chunks:
        stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            params = sig.get_Jitter(chunk, sr)
            jitters["jitter_local"].append(round(params["local"], 4))
            jitters["jitter_rap"].append(round(params["rap"], 4))
        # Ошибка может возникать при ненахождении интервалов речи
        except:
            jitters["jitter_local"].append(0)
            jitters["jitter_rap"].append(0)
        sys.stdout = stdout
    results.update(jitters)
    # Детекция эмоций
    model = AudioModel().to(device)
    model.load_state_dict(load(model_path, map_location=device))
    emotion = Torch_emotion(model, chunks=chunks, silence=silence, device=device)
    results.update({"predicts": emotion.audio_pipeline()})
    for num, val in enumerate(results["predicts"]):
        if val == "silence":
            results["Mean temp"][num] = 0
            results["Mean pitch"][num] = 0

    return results


# Дублирование парамтеров в зависимости от числа кадров в секунду
def to_fps(fps, results, output_path=False, audio_path=False, duration=1, name=False, csv=True):

    for feat in results.keys():
        suited_param = []  # Значения переведенные в соответствии с fps
        for i in results[feat]:
            # duration - длительность фрагмента (шага получения характеристик)
            suited_param.extend([i] * fps * duration)
        results[feat] = suited_param

    df = pd.DataFrame(results)
    df = dummies("predicts", df, audio_path, name)
    if csv:
        df.to_csv(output_path, index=False)
    else:
        return df


# one-hot encoding для эмоций
def dummies(col, data, audio_path=False, name=False):
    ohe = OneHotEncoder()
    emos = ["anxiety", "disgust", "happiness", "boredom", "neutral", "sadness", "anger", "silence"]
    ohe.fit(pd.DataFrame(emos))
    transform = pd.DataFrame(ohe.transform(pd.DataFrame(data[col])).toarray())
    data = data.drop(col, axis=1)
    data = pd.concat([data, transform], axis=1)
    data = data.rename(
        columns={
            0: "anxiety",
            1: "disgust",
            2: "happiness",
            3: "boredom",
            4: "neutral",
            5: "sadness",
            6: "anger",
            7: "silence",
        }
    )
    # Запись имени файла
    if name == True:
        data["name"] = audio_path
    return data


# Флаг нормализации влияет на пороговое значение тишины, флаг name влияет на запись имени файла в таблицу
# Если указать путь к видео фрагменту, создастся обработанный аудиофайл и далее операции будут производитсья с ним
def main(
    video_path=None,
    signal=None,
    audio_path=None,
    fps=1,
    normalization=True,
    name=False,
    chunk_length=1,
    sr=22050,
    csv=True,
    model_path=str(Path(__file__).parent / "weights/Torch_model.pth"),
    device="cpu",
    output_path="res.csv",
):
    # Отключение вывода в консоль (функция get_jitter выводит тип объекта)
    # sys.stdout = open(os.devnull, "w")
    if video_path is not None:
        results = features_to_timeseries(
            video_path=video_path,
            signal=None,
            model_path=model_path,
            device=device,
            chunk_length=chunk_length,
            sr=sr,
            normalization=normalization,
        )
    else:
        results = features_to_timeseries(
            audio_path=audio_path,
            model_path=model_path,
            device=device,
            chunk_length=chunk_length,
            sr=sr,
            normalization=normalization,
        )
    out = to_fps(fps=fps, results=results, audio_path=audio_path, csv=csv, output_path=output_path, name=name)

    return torch.tensor(out.to_numpy()).float()
