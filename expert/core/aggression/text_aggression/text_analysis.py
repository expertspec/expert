from __future__ import annotations

import torch
from typing import List, Tuple, Dict

from expert.core.aggression.text_aggression.text_models_en import DepreciationEN
from expert.core.aggression.text_aggression.text_models_en import ImperativeEN
from expert.core.aggression.text_aggression.text_models_en import ToxicEN

from expert.core.aggression.text_aggression.text_models_ru import DepreciationRU
from expert.core.aggression.text_aggression.text_models_ru import ImperativeRU
from expert.core.aggression.text_aggression.text_models_ru import ToxicRU


class TextAggression:
    """Extraction of aggression markers by text channel.

    Args:
        fragments (str | List[Dict]]): Transcribed speech.
        language (str, optional): Speech language for text processing ['ru', 'en']. Defaults to 'en'.
        device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.
    """

    def __init__(
        self,
        fragments: str | List[Dict],
        lang: str = "en",
        device: torch.device | None = None
    ) -> None:
        self.lang = lang
        if self.lang not in ["en", "ru"]:
            raise NotImplementedError("'lang' must be 'en' or 'ru'.")

        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        self.analysis_result = []
        self.fragments = fragments
        self.div_text_agg = []
        self.full_text_agg = []

        if type(self.fragments) not in [str, list]:
            raise ValueError("'fragments' must be 'str' or 'list'.")

    def is_imperative(self) -> List[bool]:
        if self.lang == "en":
            imperative = ImperativeEN()

            if type(self.fragments) is str:
                return [imperative.is_imperative(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    prediction = imperative.is_imperative(fragment["text"])
                    fragment["is_imperative"] = prediction
                    result_list.append(prediction)
                return result_list
            else:
                raise ValueError("Input sents must be 'str' or 'list'.")
        elif self.lang == "ru":
            imperative = ImperativeRU()
            if type(self.fragments) is str:
                return [imperative.is_imperative(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    prediction = imperative.is_imperative(fragment["text"])
                    fragment["is_imperative"] = prediction
                    result_list.append(prediction)
                return result_list

    def is_toxic(self) -> List[bool]:
        if self.lang == "en":
            toxic = ToxicEN(device=self._device)
            if type(self.fragments) is str:
                return [toxic.is_toxic(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    prediction = toxic.is_toxic(fragment["text"])
                    fragment["is_toxic"] = prediction
                    result_list.append(prediction)
                return result_list

        elif self.lang == "ru":
            toxic = ToxicRU(self._device)
            if type(self.fragments) is str:
                return [toxic.is_toxic(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    prediction = toxic.is_toxic(fragment["text"])
                    fragment["is_toxic"] = prediction
                    result_list.append(prediction)
                return result_list

    def is_depreciation(self) -> int:
        if self.lang == "en":
            depreciation = DepreciationEN()
            if type(self.fragments) is str:
                return [depreciation.is_depreciation(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    prediction = depreciation.is_depreciation(fragment["text"])
                    fragment["is_deprication"] = prediction
                    result_list.append(prediction)
                return result_list
        elif self.lang == "ru":
            depreciation = DepreciationRU()
            if type(self.fragments) is str:
                return [len(depreciation.is_depreciation(self.fragments)[1])]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    prediction = len(
                        depreciation.is_depreciation(fragment["text"])[1])
                    fragment["is_deprication"] = prediction
                    result_list.append(prediction)
                return result_list

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def get_report(self) -> Tuple[List, List]:
        self.analysis_result.append(self.is_depreciation())
        self.analysis_result.append(self.is_imperative())
        self.analysis_result.append(self.is_toxic())

        toxic_part = 0
        depreciation_part = 0
        imperative_part = 0

        for num in range(len(self.analysis_result[0])):
            if self.analysis_result[0][num] > 0:
                depreciation_part += 1
            if self.analysis_result[1][num]:
                imperative_part += 1
            if self.analysis_result[2][num]:
                toxic_part += 1

        self.div_text_agg = self.fragments
        self.full_text_agg = {
            "toxic_part": toxic_part / len(self.fragments),
            "depreciation_part": depreciation_part / len(self.fragments),
            "imperative_part": imperative_part / len(self.fragments),
        }

        return (
            self.div_text_agg,
            self.full_text_agg
        )
