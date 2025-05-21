import multiprocessing
from tqdm import tqdm
import time
import __main__
from typing import (
    List,
    Set,
    Optional,
    Dict,
    Union,
)
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import (
    dataclass,
)
import os
import sys

sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))
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
    DUSHA_WAVS_DIR_NAME,
    PROCESSED_DUSHA_CROWD_TRAIN_DIR_PATH,
    PROCESSED_DUSHA_CROWD_TEST_DIR_PATH,
)

from configs.base import (
    RB_OPEN_FILE_MODE,
    RUSSIAN_VOWELS,
    TAB,
    DROP_DUPLICATES_KEEP_FIRST,
    EMPTY,
)
from high_level_feature_extractor.volume.human_speech import (
    HIGH_FREQUENCY_SPEECH_THRESHOLD,
)
from configs.paths import (
    DUSHA_CROWD_TRAIN_FILE_PATH,
)
from utils.parallel_processing import (
    divide_into_chunks,
)
from configs.datasets.dusha import (
    HASH_ID_COLUMN_NAME,
)
from models.asr.whisper import (
    PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_WHISPER_LARGE_V3_FILE_PATH,
)
from models.asr.transcribe import (
    TRANSCRIPTION_COLUMN_NAME,
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
    hash_2_transcription:Dict[str, str],
    chunks_quantity:int = 10,
    rows_quantity:Optional[int] = None,
    output_file_path:Path = PROCESSED_DUSHA_CROWD_TRAIN_HLF_LAST_VERSION_FILE_PATH,
    reading_mode: str = RB_OPEN_FILE_MODE,
    hash_col_name:str = HASH_ID_COLUMN_NAME,
    drop_duplicates_keep:str = DROP_DUPLICATES_KEEP_FIRST,
    # HF_threshold: int = HIGH_FREQUENCY_SPEECH_THRESHOLD,
    # vowels:Set[str] = RUSSIAN_VOWELS,
    )->pd.Series:
    unique_hashes:np.ndarray = df.hash_id.unique()
    print(f'len(unique_hashes) = {len(unique_hashes)}')
    if rows_quantity is not None:
        unique_hashes = unique_hashes[:rows_quantity]
    print('Start of processing arguments_list!')
    arguments_list:List[WAVFilePathInitArgs] = []
    df = df.drop_duplicates(subset=hash_col_name, keep=drop_duplicates_keep)
    for _, row in df.iterrows():

        # hash_id and speaker_text do not depends on annotator answer, mb do drop duplicates by hash_id before
        file_path:Path = wavs_dir_path / Path(row.audio_path).name
        transcription:Union[str, float] = hash_2_transcription[row.hash_id]
        arguments:WAVFilePathInitArgs = WAVFilePathInitArgs(
            path=file_path, 
            transcription=transcription if isinstance(transcription, str) else EMPTY, 
            reading_mode=reading_mode,
        )
        arguments_list.append(arguments)
    
    # for hash in unique_hashes:
    #     # hash_id and speaker_text do not depends on annotator answer, mb do drop duplicates by hash_id before
    #     row:pd.Series = df[df.hash_id == hash].iloc[0]
    #     file_path:Path = wavs_dir_path / Path(row.audio_path).name
        
    #     arguments:WAVFilePathInitArgs = WAVFilePathInitArgs(
    #         path=file_path, 
    #         transcription=row.speaker_text, 
    #         reading_mode=reading_mode,
    #     )
    #     arguments_list.append(arguments)
    print(f'End of processing arguments_list, len(arguments_list) = {len(arguments_list)}!')
    results:List[HashHLF] = []

    arguments_chunks:List[List[WAVFilePathInitArgs]] = divide_into_chunks(
        lst=arguments_list,
        k=chunks_quantity,
    )
    for arguments_chunk in tqdm(arguments_chunks):

        with multiprocessing.Pool(processes=num_processes) as pool:
            
            results_chunk:List[HashHLF] = pool.map(AudioWAVFilePathInitArgs_2_HashHLF, arguments_chunk)
            results_chunk = list(results_chunk)
            results += results_chunk

        series:pd.Series = pd.Series(map(lambda el: repr(el), results))

        series.to_csv(output_file_path, index=False, header=False)
    return series

@dataclass
class RawCrowdHLFExtractingPaths:
    crowd_file_path:Path
    wavs_dir_path:Path
    output_file_path:Path

def extract(
    num_processes:int = 40,
    chunks_quantity:int = 40,
    ):
    hash_2_transcription:Dict[str, Union[str, float]] = dict(
        pd.read_csv(
            PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_WHISPER_LARGE_V3_FILE_PATH, 
            index_col=0,
        )[TRANSCRIPTION_COLUMN_NAME]
    )
    print('Start of the processing!')

    train_paths:RawCrowdHLFExtractingPaths = RawCrowdHLFExtractingPaths(
        crowd_file_path=DUSHA_CROWD_TRAIN_FILE_PATH,
        wavs_dir_path=PROCESSED_DUSHA_CROWD_TRAIN_DIR_PATH / DUSHA_WAVS_DIR_NAME,
        output_file_path=PROCESSED_DUSHA_CROWD_TRAIN_HLF_LAST_VERSION_FILE_PATH,
    )
    test_paths:RawCrowdHLFExtractingPaths = RawCrowdHLFExtractingPaths(
        crowd_file_path=DUSHA_CROWD_TEST_FILE_PATH,
        wavs_dir_path=PROCESSED_DUSHA_CROWD_TEST_DIR_PATH / DUSHA_WAVS_DIR_NAME,
        output_file_path=PROCESSED_DUSHA_CROWD_TEST_HLF_LAST_VERSION_FILE_PATH,
    )

    for paths in [train_paths, test_paths]:
        raw_crowd:pd.DataFrame = pd.read_csv(paths.crowd_file_path, sep=TAB)
        raw_crowd_2_HLF(
            df=raw_crowd,
            wavs_dir_path=paths.wavs_dir_path,
            num_processes=num_processes,
            chunks_quantity=chunks_quantity,
            hash_2_transcription=hash_2_transcription,
            # rows_quantity=100,
            output_file_path=paths.output_file_path,
        )

if __name__ == __main__.__name__:  
    extract()

# Run script:
# nohup python extract.py > processes_outputs/HLF_extracting.out & disown
