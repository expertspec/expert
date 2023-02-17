from __future__ import annotations

import json
import pandas as pd
from typing import List, Tuple, Union
import torch

# Local dependencies:
from expert.core.aggression.text_aggression.text_models_en import Depreciation as DepreciationEn
from expert.core.aggression.text_aggression.text_models_en import Imperative as ImperativeEn
from expert.core.aggression.text_aggression.text_models_en import Toxic as ToxicEn

from expert.core.aggression.text_aggression.text_models_ru import Depreciation as DepreciationRu
from expert.core.aggression.text_aggression.text_models_ru import Imperative as ImperiveRu
from expert.core.aggression.text_aggression.text_models_ru import Toxic as ToxicRu


class TextAggression:
    """Aggression markers extraction across the text channel."""
    def __init__(
        self,
        # words_path
        fragments: Union[str, List[str]],
        lang: str = 'en',
        device: torch.device | None = None
    ) -> None:
        if lang not in ['en', 'ru']:
            raise NameError
        self._device = torch.device("cpu")
        
        if device is not None:
            self._device = device
        self.lang = lang
        self.analysis_result = []
        self.fragments = fragments
        self.div_aud_agg = []
        if type(self.fragments) not in [str, list]:
            raise ValueError("Fragments should be str or list")

    def is_imperative(self) -> List[bool]:
        if self.lang == 'en':
            imperative = ImperativeEn()

            if type(self.fragments) is str:
                return [imperative.is_imperative(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    result_list.append(imperative.is_imperative(fragment))
                return result_list
            else:
                raise ValueError('Input sents must be str or list')
        elif self.lang == 'ru':
            imperative = ImperiveRu()
            if type(self.fragments) is str:
                return [imperative.is_imperative(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    result_list.append(imperative.is_imperative(fragment))
                return result_list

    def is_toxic(self) -> List[bool]:
        if self.lang == 'en':
            toxic = ToxicEn(self._device)
            if type(self.fragments) is str:
                return [toxic.is_toxic(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for sent in self.fragments:
                    result_list.append(toxic.is_toxic(sent))
                return result_list

        elif self.lang == 'ru':
            toxic = ToxicRu(self._device)
            if type(self.fragments) is str:
                return [toxic.is_toxic(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    result_list.append(toxic.is_toxic(fragment))
                return result_list

    def is_depreciation(self) -> int:
        if self.lang == 'en':
            depreciation = DepreciationEn()
            if type(self.fragments) is str:
                return [depreciation.is_deprication(self.fragments)]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    result_list.append(depreciation.is_deprication(fragment))
                return result_list
        elif self.lang == 'ru':
            depreciation = DepreciationRu()
            if type(self.fragments) is str:
                return [len(depreciation.is_depreciation(self.fragments)[1])]
            elif type(self.fragments) is list:
                result_list = []
                for fragment in self.fragments:
                    result_list.append(len(depreciation.is_depreciation(fragment)[1]))
                return result_list

    @property
    def device(self) -> torch.device:
        """Check the device type."""
        return self._device


    def get_report(self):

        self.analysis_result.append(self.is_depreciation())
        self.analysis_result.append(self.is_imperative())
        self.analysis_result.append(self.is_toxic())

        toxic_part = 0
        deprecation_part = 0
        imperative_part = 0

        for num in range(len(self.analysis_result[0])):
            self.div_aud_agg.append({'num_deprecation': self.analysis_result[0][num],
                                    'is_imperative': self.analysis_result[1][num],
                                     'is_toxic': self.analysis_result[2][num]})
            if self.analysis_result[0][num] > 0:
                deprecation_part += 1
            if self.analysis_result[1][num]:
                imperative_part += 1
            if self.analysis_result[2][num]:
                toxic_part += 1
        self.full_aud_agg = {'toxic_part': toxic_part / len(self.div_aud_agg),
                      'imperative_part': imperative_part / len(self.div_aud_agg),
                      'deprecation_part': deprecation_part / len(self.div_aud_agg)
                      }

        return (
            self.div_aud_agg,
            self.full_aud_agg
        )