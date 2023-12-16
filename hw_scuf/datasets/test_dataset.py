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



class TestSet(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):

        self.kaggle = True
        if data_dir is None:
            data_dir = ROOT_PATH / "test_data"
            data_dir.mkdir(exist_ok=True, parents=True)
        elif isinstance(data_dir, str):
            data_dir = Path(data_dir)
            self.kaggle = True

        self._data_dir = data_dir
        index = self._get_or_load_index()
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self._data_dir / "test_index.json" 
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        for f in self._data_dir.iterdir():
            index.append(
                    {
                        "path": str(f.absolute()),
                        "target": True
                    }
                )
        logger.info('Found ' + str(len(index)) + ' flac files')
        return index
