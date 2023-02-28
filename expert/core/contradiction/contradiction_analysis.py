from __future__ import annotations

import json
import os
from typing import List, Union

import torch

from expert.core.contradiction.contr_tools import model_tools


class ContradictionDetector:
    """Class for working with detector of contradiction"""
    def __init__(
        self,
        transcription_path: str,
        path_to_video: str,
        lang: str = 'en',
        device: torch.device | None = None,
        save_to: str = "app\\temp",
        chosen_intervals: list[int] = [],
        interval_duration: int = 60,
    ) -> None:
        """Object initialization.
         Creates instance with information about language and device

        Args:
            transcription_path (str): path to json file with words from transcribation
            path_to_video (str): path to the video
            lang (str, optional): flag of the language can be 'en'|'ru'. Defaults to 'en'.
            device (str, optional): setting of computational device 'cpu'|'cuda'. Defaults to 'cpu'.
            save_to (str): Path to save JSON with information about contradiction.
            chosen_intervals (list[int], optional): flags for intervals for analyzis. Defaults to [].
            For default settings will be analyzed full text
            interval_duration (int, optional): the length of intervals (value in seconds). Defaults to 60.
        """
        self.lang = lang

        self._device = torch.device("cpu")
        self.model = model_tools.create_model(self.lang, self._device)

        if device is not None:
            self._device = device
            self.model.to(self._device)

        self.transcription_path = transcription_path
        self.path_to_video = path_to_video
        self.save_to = save_to
        self.chosen_intervals = chosen_intervals
        self.interval_duration = interval_duration
    
    def get_sentences(self, all_words):
        sentences = []
        if all_words:
            end = all_words[0]['end']
            sentence = all_words[0]['word']
            
            for idx in range(len(all_words[:-1])):
                if all_words[idx+1]['start'] - end < 1.0:
                    sentence += ' ' + all_words[idx+1]['word']
                    end = all_words[idx+1]['end']
                else:
                    sentences.append(sentence)
                    end = all_words[idx+1]['end']
                    sentence = all_words[idx+1]['word']
            sentences.append(sentence)
        else:
            raise 'No words'
        return sentences
        
    def get_phrases(self, all_words, duration=60) -> List[dict]:
        phrases = []

        assert len(all_words) > 1, "not enough words"

        while all_words:
            init_elem = all_words.pop(0)
            phrase = init_elem["word"]
            time_left = duration - (init_elem["end"] - init_elem["start"])
            end_time = init_elem["end"]
            while time_left > 0 and all_words:
                elem = all_words.pop(0)
                phrase = phrase + " " + elem["word"]
                time_left -= elem["end"] - end_time
                end_time = elem["end"]
            else:
                phrases.append({"time": [init_elem["start"], elem["end"]], "text": phrase})
        return phrases

    def analysis(self,
                entered_text: str,
                texts: Union[str, List[str]],
                ind=0
                ) -> List[dict]:
        analysis_results = []
        if isinstance(texts, str):
            part, predict = model_tools.predict_inference(
                entered_text, texts, self.model, lang=self.lang, device=self._device
            )
            analysis_results.append(
                {
                    "text": part.replace(" [SEP]", ""),
                    "predict": float(predict),
                    "interval": ind,
                }
            )
        elif isinstance(texts, list):
            for text in texts:
                part, predict = model_tools.predict_inference(
                entered_text, text, self.model, lang=self.lang, device=self._device
            )
                try:
                    analysis_results.append(
                        {
                            "text": part.replace(" [SEP]", ""),
                            "predict": float(predict),
                            "interval": ind,
                        }
                    )
                except AttributeError:
                    analysis_results.append(
                        {
                            "text": part[0].replace(" [SEP]", ""),
                            "predict": float(predict),
                            "interval": ind,
                        }
                    )
        else:
            raise TypeError
        return analysis_results

    @property
    def device(self) -> torch.device:
        """Check the device type."""
        return self._device

    def get_contradiction(self,
                        entered_text: str,
                        ) -> None:
        """Function for text analyzing.
        Creates json file with predictions

        Args:
            entered_text (str): utterance for analyzis
        """        
        with open(self.transcription_path, 'r') as f:
            words = json.load(f)
        
        if len(self.chosen_intervals):
            fragments = self.get_phrases(words[:], duration=self.interval_duration)
            intervals = dict.fromkeys(range(len(fragments)))
            for interval in self.chosen_intervals:
                interval_words = []
                for word in words:
                    if fragments[interval]["time"][0] <= word['start'] and word['end'] <= fragments[interval]["time"][1]:
                        interval_words.append(word)
                texts = self.get_sentences(interval_words)
                intervals[interval] = texts
            
            contr_data = []

            for ind in intervals:
                if intervals[ind]:
                    analysis_results = self.analysis(entered_text, texts, ind=ind)
                    contr_data.append(analysis_results)
            
            # Unpacking of the list of lists
            contr_data = [item for sublist in contr_data for item in sublist]
        
        else:
            texts = self.get_sentences(words)
            contr_data = self.analysis(entered_text, texts)

        temp_path = os.path.splitext(os.path.basename(self.path_to_video))[0]
        if not os.path.exists(os.path.join(self.save_to, temp_path)):
            os.makedirs(os.path.join(self.save_to, temp_path))
            
        with open(os.path.join(self.save_to, temp_path, 'contradiction_report.json'), 'w') as filename:
            json.dump(contr_data, filename)
        
        return contr_data

