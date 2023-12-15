import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_scuf.base.base_dataset import BaseDataset
from hw_scuf.utils import ROOT_PATH

logger = logging.getLogger(__name__)



class ASV(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):

        self.kaggle = True
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        elif isinstance(data_dir, str):
            data_dir = Path(data_dir)
            self.kaggle = True

        self._data_dir = data_dir
        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
       
        if self.kaggle:
            Path('kaggle/datasets/asv').mkdir(exist_ok=True, parents=True)
            index_path = Path('kaggle/datasets/asv') / f"{part}_index.json"
        else:
            index_path = self._data_dir / f"{part}_index.json" 
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        tg_dir = self._data_dir / 'ASVspoof2019_LA_cm_protocols'
        index = []
        if part == 'train':
            split_dir = self._data_dir / 'ASVspoof2019_LA_train' / 'flac'
            tg_file = tg_dir / 'ASVspoof2019.LA.cm.train.trn.txt'
        elif part == 'dev':
            split_dir = self._data_dir / 'ASVspoof2019_LA_dev' / 'flac'
            tg_file = tg_dir / 'ASVspoof2019.LA.cm.dev.trl.txt'
        elif part == 'eval':
            split_dir = self._data_dir / 'ASVspoof2019_LA_eval' / 'flac'
            tg_file = tg_dir / 'ASVspoof2019.LA.cm.eval.trl.txt'
        else:
            assert False, "Invalid part"
        target = dict()
        with tg_file.open() as f:
            for line in f:
                data = line.split()
                target[data[1]] = (data[-1] == 'spoof')
        assert split_dir.exists(), "No data folder"

        for file in tqdm(split_dir.iterdir(), desc='making index'):
            if file.stem in target:
                index.append(
                    {
                        "path": str(file.absolute()),
                        "target": target[file.stem]
                    }
                )
            else:
                logger.info('File ' + file.stem + ' has no target')
        logger.info('Found ' + str(len(index)) + ' flac files')
        return index
