from dataclasses import (
    dataclass,
    asdict,
)
from typing import (
    List,
    Optional,
)
from pathlib import (
    Path,
)
import __main__
import multiprocessing
import pandas as pd
import os
import sys
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

from models.data_preparation import (
    get_VQEd_audio_file_paths,
    ProcessFilePathsChunkArgs,
)
from models.bio.wav2vec2_large_robust.model import (
    Wav2VecBioModel,
    Wav2VecBioModelPredProba,
    MODEL_NEEDED_RAM,
)
from utils.parallel_processing import (
    divide_into_chunks,
    get_available_gpu_indices_with_free_memory,
)
from configs.paths import (
    # TODO: set paths in more local files
    PROCESSED_DUSHA_CROWD_BIO_DIR_PATH,
    W2V_BIO_DIR_PATH,
)
from configs.base import (
    DOT_CSV,
)
from configs.datasets.dusha import (
    SAMPLE_RATE,
)
from utils.dataclass import (
    flatten_dict,
)

PROCESSED_DUSHA_CROWD_W2V_BIO_FILE_PATH:Path = PROCESSED_DUSHA_CROWD_BIO_DIR_PATH / (Path(W2V_BIO_DIR_PATH).name + DOT_CSV)

@dataclass
class PredictOutput:
    hash_id:str
    bio_proba:Wav2VecBioModelPredProba

def predict_chunk(
    arguments:ProcessFilePathsChunkArgs,
    ) -> List[PredictOutput]:
    model:Wav2VecBioModel = Wav2VecBioModel.dir_path_with_device_init(
        device_name=arguments.gpu_formatted_keyword(),
    )
    processed:List[PredictOutput] = []
    for wav_path in arguments.file_paths:
        predict:Wav2VecBioModelPredProba =  model.audio_file_path_2_predict(
            path=wav_path,
            sample_rate=SAMPLE_RATE,
        )
        processed.append(
            PredictOutput(
                hash_id=wav_path.stem,
                bio_proba=predict,
            )
        )
    
    return processed

if __name__ == __main__.__name__:
    multiprocessing.set_start_method('spawn')
    wavs_paths_to_process_limit:Optional[int] = None # 10000
    max_processes_quantity:int = 40
    chunks_quantity:int = 100
    
    wavs_paths_to_process:List[Path] = get_VQEd_audio_file_paths()[:wavs_paths_to_process_limit]
    print(f'len(wavs_paths_to_transribe) = {len(wavs_paths_to_process)}')

    output_file_path:Path = PROCESSED_DUSHA_CROWD_W2V_BIO_FILE_PATH

    arguments_chunks:List[List[Path]] = divide_into_chunks(
        lst=wavs_paths_to_process[:],
        k=chunks_quantity,
    )

    processed:List[PredictOutput] = []
    for arguments_chunk in arguments_chunks:
        gpu_indices_with_free_memory:List[int] = get_available_gpu_indices_with_free_memory(model_needed_ram=MODEL_NEEDED_RAM)
        # print(f'gpu_indices_with_free_memory = {gpu_indices_with_free_memory}')
        num_processes:int = min(len(gpu_indices_with_free_memory), max_processes_quantity)
        # print(f'num_processes = {num_processes}')
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
            processed_chunk_chunks:List[List[PredictOutput]] = pool.map(predict_chunk, process_arguments_list)
        processed_chunk:List[PredictOutput] = []
        for processed_chunk_chunk in processed_chunk_chunks:
            processed_chunk.extend(processed_chunk_chunk)
        
        # print(f'len(processed_chunk) = {len(processed_chunk)}')
        processed.extend(processed_chunk)
        # rewrite to jsonl with appending, do not use processed
        df:pd.DataFrame = pd.DataFrame(list(map(lambda x: asdict(x), processed)))
        df.to_csv(output_file_path, index=False, header=True)
