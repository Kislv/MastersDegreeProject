from pathlib import Path
from typing import(
    List,
    Optional,
    Dict, 
)
import os
import __main__
import sys
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
import multiprocessing
import pynvml
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))


from configs.datasets.dusha import (
    HASH_ID_COLUMN_NAME,
)
from configs.paths import (
    PROCESSED_DUSHA_CROWD_TRAIN_DIR_PATH,
    PROCESSED_DUSHA_CROWD_TEST_DIR_PATH,
    DUSHA_WAVS_DIR_NAME,
    PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_DIR_PATH,
)
from configs.base import (
    DOT_CSV,
    SPACE,
)
from models.asr.whisper import (
    Whisper,
    WHISPER_LARGE_MODEL_NEEDED_RAM,
    WHISPER_LARGE_V3_MODEL_NAME as MODEL_NAME,
    PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_WHISPER_LARGE_V3_FILE_PATH,
)
from models.data_preparation import (
    get_VQEd_audio_file_paths,
    ProcessFilePathsChunkArgs,
)
from utils.parallel_processing import (
    CUDA_WITH_INDEX_TEMPLATE,
    divide_into_chunks,
    get_available_gpu_indices_with_free_memory,
)

TRANSCRIPTION_COLUMN_NAME:str = 'transcription'


@dataclass
class TranscribeOutput:
    hash_id:str
    transcription:str

def transcribe_chunk(
    arguments:ProcessFilePathsChunkArgs,
    transcription_sep:str = SPACE,
    ) -> List[TranscribeOutput]:
    model:Whisper = Whisper.device_name_dir_path_init(
        device_name=CUDA_WITH_INDEX_TEMPLATE.format(index=arguments.gpu_index),
    )
    processed:List[TranscribeOutput] = []
    for wav_path in arguments.file_paths:
        transcription:str =  model.audio_file_2_transcription(
            audio_path=wav_path,
        )
        processed.append(
            TranscribeOutput(
                hash_id=wav_path.stem,
                transcription=transcription,
            )
        )
    
    return processed

if __name__ == __main__.__name__:
    multiprocessing.set_start_method('spawn')
    wavs_paths_to_transribe:List[Path] = get_VQEd_audio_file_paths()
    print(f'len(wavs_paths_to_transribe) = {len(wavs_paths_to_transribe)}')

    output_file_path:Path = PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_WHISPER_LARGE_V3_FILE_PATH

    chunks_quantity:int = 100
    
    arguments_chunks:List[List[Path]] = divide_into_chunks(
        lst=wavs_paths_to_transribe[:],
        k=chunks_quantity,
    )

    processed:List[TranscribeOutput] = []
    for arguments_chunk in arguments_chunks:
        gpu_indices_with_free_memory = get_available_gpu_indices_with_free_memory(model_needed_ram=WHISPER_LARGE_MODEL_NEEDED_RAM)
        num_processes:int = len(gpu_indices_with_free_memory)

        arguments_chunk_chunks:List[List[Path]] = divide_into_chunks(
            lst=arguments_chunk[:],
            k=num_processes,
        )
        # print(f'len(arguments_chunk_chunks) = {len(arguments_chunk_chunks)}')
        process_arguments_list:List[ProcessFilePathsChunkArgs] = list(
            map(
                lambda i: ProcessFilePathsChunkArgs(
                    gpu_index=gpu_indices_with_free_memory[i],
                    file_paths=arguments_chunk_chunks[i],
                ),
                range(len(arguments_chunk_chunks)),
            )
        )
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            processed_chunk_chunks:List[List[TranscribeOutput]] = pool.map(transcribe_chunk, process_arguments_list)
        processed_chunk:List[TranscribeOutput] = []
        for processed_chunk_chunk in processed_chunk_chunks:
            processed_chunk.extend(processed_chunk_chunk)
        
        # print(f'len(processed_chunk) = {len(processed_chunk)}')
        processed.extend(processed_chunk)
        # rewrite to jsonl with appending, do not use processed
        df:pd.DataFrame = pd.DataFrame(processed)
        df.to_csv(output_file_path, index=False, header=True)
