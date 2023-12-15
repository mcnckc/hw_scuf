import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_scuf.base.base_text_encoder import BaseTextEncoder
from hw_scuf.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            max_spec_len=600,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None,
            max_text_length=None,
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.log_spec = config_parser["preprocessing"]["log_spec"]
        self.max_spec_len = max_spec_len
        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)
        audio_wave, audio_spec = self.process_wave(audio_wave)
        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
            "target": data_dict['target']
        }

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)

            if audio_tensor_spec.shape[-1] < self.max_spec_len:
                a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=float)
                torch.nn.functional.pad(a[None, ...], (0, 2), mode='circular').squeeze(dim=0)
                print("OK,", audio_tensor_spec.shape)
                audio_tensor_spec = torch.nn.functional.pad(audio_tensor_spec[None, ...], 
                                        (0, self.max_spec_len - audio_tensor_spec.shape[-1]),
                                        'circular').squeeze(dim=0)
            else:
                audio_tensor_spec = audio_tensor_spec[:, :self.max_spec_len]

            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(
            index: list, limit
    ) -> list:
        initial_size = len(index)
        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
