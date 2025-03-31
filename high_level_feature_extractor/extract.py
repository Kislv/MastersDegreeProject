import multiprocessing
from tqdm import tqdm
import time
import __main__
from typing import (
    List,
    Set,
    Optional,
)
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('..')

from audio import (
    Audio,
    WAVFilePathInitArgs,
)

from high_level_feature_extractor.extractor import (
    HighLevelSpeechFeatures,
    HashHLF,
)

from configs.paths import (
    DUSHA_CROWD_TEST_FILE_PATH,
    DUSHA_CROWD_TRAIN_WAVS_DIR_PATH,
    DUSHA_CROWD_TEST_WAVS_DIR_PATH,
    PROCESSED_DUSHA_CROWD_TRAIN_HLF_LAST_VERSION_FILE_PATH,
    PROCESSED_DUSHA_CROWD_TEST_HLF_LAST_VERSION_FILE_PATH,
)

from configs.base import (
    RB_OPEN_FILE_MODE,
    RUSSIAN_VOWELS,
    TAB,
)
from volume.human_speech import (
    HIGH_FREQUENCY_SPEECH_THRESHOLD,
)
from configs.paths import (
    DUSHA_CROWD_TRAIN_FILE_PATH,
)
from utils.parallel_processing import (
    divide_into_chunks,
)

def AudioWAVFilePathInitArgs_2_HashHLF(
    arguments:WAVFilePathInitArgs,
    )->Optional[HashHLF]:
    # print(f'audio.hash = {audio.hash}')
    # time.sleep(1)
    try:
        audio:Audio = Audio.wav_file_path_init(arguments=arguments)
        HLF:HighLevelSpeechFeatures = HighLevelSpeechFeatures.audio_init(
            audio=audio,
        )
        return HashHLF(
            hash=audio.hash, 
            features=HLF,
        )
    except Exception as e:
        print(f'Exception = {e}, arguments = {arguments}')

def raw_crowd_2_HLF(
    # TODO: dataclass raw_crowd
    df:pd.DataFrame,
    wavs_dir_path:Path,
    num_processes:int,
    chunks_quantity:int = 10,
    rows_quantity:Optional[int] = None,
    output_file_path:Path = PROCESSED_DUSHA_CROWD_TRAIN_HLF_LAST_VERSION_FILE_PATH,
    reading_mode: str = RB_OPEN_FILE_MODE,
    HF_threshold: int = HIGH_FREQUENCY_SPEECH_THRESHOLD,
    vowels:Set[str] = RUSSIAN_VOWELS,
    )->pd.Series:
    unique_hashes:np.ndarray = df.hash_id.unique()
    print(f'len(unique_hashes) = {len(unique_hashes)}')
    if rows_quantity is not None:
        unique_hashes = unique_hashes[:rows_quantity]
    
    arguments_list:List[WAVFilePathInitArgs] = []
    for hash in unique_hashes:
        # hash_id and speaker_text do not depends on annotator answer, mb do drop duplicates by hash_id before
        row:pd.Series = df[df.hash_id == hash].iloc[0]
        file_path:Path = wavs_dir_path / Path(row.audio_path).name
        
        arguments:WAVFilePathInitArgs = WAVFilePathInitArgs(
            path=file_path, 
            transcription=row.speaker_text, 
            reading_mode=reading_mode,
        )
        arguments_list.append(arguments)
    
    results:List[HashHLF] = []

    arguments_chunks:List[List[WAVFilePathInitArgs]] = divide_into_chunks(
        lst=arguments_list,
        k=chunks_quantity,
    )
    for arguments_chunk in tqdm(arguments_chunks):

        with multiprocessing.Pool(processes=num_processes) as pool:
            
            results_chunk:List[HashHLF] = list(pool.map(AudioWAVFilePathInitArgs_2_HashHLF, arguments_chunk))
            results += results_chunk

        series:pd.Series = pd.Series(map(lambda el: repr(el), results))

        series.to_csv(output_file_path, index=False, header=False)
    return series



if __name__ == __main__.__name__:  
    raw_crowd:pd.DataFrame = pd.read_csv(DUSHA_CROWD_TEST_FILE_PATH, sep=TAB)
    raw_crowd_2_HLF(
        df=raw_crowd,
        wavs_dir_path=DUSHA_CROWD_TEST_WAVS_DIR_PATH,
        num_processes=20,
        chunks_quantity=10,
        # rows_quantity=100,
        output_file_path=PROCESSED_DUSHA_CROWD_TEST_HLF_LAST_VERSION_FILE_PATH,
    )

# Run script:
# nohup python extract.py & disown
