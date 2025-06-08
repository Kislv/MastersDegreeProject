import os
from typing import (
    List,
)
from pathlib import (
    Path,
)
from dataclasses import (
    dataclass,
)
from configs.paths import (
    PROCESSED_DUSHA_CROWD_TRAIN_DIR_PATH,
    PROCESSED_DUSHA_CROWD_TEST_DIR_PATH,
    DUSHA_WAVS_DIR_NAME,
    PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_DIR_PATH,
)
from utils.parallel_processing import (
    CUDA_WITH_INDEX_TEMPLATE,
)

def get_VQEd_audio_file_paths(
    wavs_dir_name:str = DUSHA_WAVS_DIR_NAME,
    processed_dusha_crowd_part_dir_paths:List[Path] = [PROCESSED_DUSHA_CROWD_TRAIN_DIR_PATH, PROCESSED_DUSHA_CROWD_TEST_DIR_PATH],
    ) -> List[Path]:
    VQEd_wavs_dirs:List[Path] = list(map(lambda x: x / wavs_dir_name, processed_dusha_crowd_part_dir_paths))
    wavs_to_transriber:List[Path] = []
    for wavs_dir in VQEd_wavs_dirs:
        wavs_to_transriber.extend(list(map(lambda x: wavs_dir / x, os.listdir(wavs_dir))))
    return wavs_to_transriber

@dataclass
class ProcessFilePathsChunkArgs:
    gpu_index:int
    file_paths:List[Path]
    def gpu_formatted_keyword(
        self,
        template:str = CUDA_WITH_INDEX_TEMPLATE,
        )->str:
        return template.format(index=self.gpu_index)
