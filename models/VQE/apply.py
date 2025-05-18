from typing import (
    List,
    Dict,
    Callable,
    Set,
)
from dataclasses import dataclass
import multiprocessing
from tqdm import tqdm
import __main__
from pathlib import Path
import os
import sys
sys.path.append(os.getenv('MASTER_DEPLOMA_PROJECT_FILE_PATH'))

from configs.paths import (
    SEPFORMER_16KHZ_DIR_PATH,
)
from models.VQE.sepformer import (
    VQEArgs,
    Sepformer,
)
from utils.parallel_processing import (
    divide_into_chunks,
)
from configs.paths import PART_NAME_2_WAVS_DIR_PATH, PART_NAME_2_PROCESSED_DUSHA_CROWD_DIR_PATH, DUSHA_WAVS_DIR_NAME, DATASET_PARTS_NAMES

def get_VQE_args_list(
    parts_names:List[str] = DATASET_PARTS_NAMES,
    part_name_2_wavs_dir_path:Dict[str, Path] = PART_NAME_2_WAVS_DIR_PATH,
    part_name_2_processed_dusha_crowd_dir_path:Dict[str, Path] = PART_NAME_2_PROCESSED_DUSHA_CROWD_DIR_PATH,
    wavs_dir_name:str = DUSHA_WAVS_DIR_NAME,
    )->List[VQEArgs]:
    VQE_args_list:List[VQEArgs] = []
    for part_name in parts_names:
        # display(raw_crowd.head(2))
        # hashs = raw_crowd[HASH_ID_COLUMN_NAME].unique()
        # audio_paths:np.ndarray = raw_crowd[AUDIO_PATH_FIELD_NAME].unique()
        
        input_dir_path:Path = part_name_2_wavs_dir_path[part_name]
        output_dir_path:Path = part_name_2_processed_dusha_crowd_dir_path[part_name] / wavs_dir_name
        output_dir_path.mkdir(exist_ok=True)
        input_file_names:Set[str] = set(os.listdir(input_dir_path))
        output_file_names:Set[str] = set(os.listdir(output_dir_path))
        left_files_2_prcess:List[str] = list(input_file_names.difference(output_file_names))
        
        print(f'len(left_files_2_prcess) = {len(left_files_2_prcess)}')
        VQE_args_list.extend(
            list(
                map(
                    lambda file_name: VQEArgs(
                        input_file_path=input_dir_path / file_name, 
                        output_file_path=output_dir_path / file_name,
                    ), 
                    left_files_2_prcess,
                )
            )
        )
        # for input_file_name in input_file_names:
            
    print(f'len(VQE_args_list) = {len(VQE_args_list)}')
    return VQE_args_list

@dataclass
class VQE_process_chunk_args:
    arguments_list:List[VQEArgs]
    model_file_path:Path = SEPFORMER_16KHZ_DIR_PATH

def apply_VQE_to_args_list(
    arguments:VQE_process_chunk_args,
    ):
    sepforme_16khz:Sepformer = Sepformer.file_init(
        path=arguments.model_file_path,
    )
    apply_VQE:Callable = sepforme_16khz.apply_VQE
    for arguments in arguments.arguments_list:
        apply_VQE(arguments=arguments)

if __name__ == __main__.__name__:  
    multiprocessing.set_start_method('spawn')
    VQE_args_list:List[VQEArgs] = get_VQE_args_list()

    # sepforme_16khz:Sepformer = Sepformer.file_init(
    #     path=SEPFORMER_16KHZ_DIR_PATH,
    # )
    # apply_VQE:Callable = sepforme_16khz.apply_VQE
    chunks_quantity:int = 10
    num_processes:int = 10
    arguments_chunks:List[List[VQEArgs]] = divide_into_chunks(
        lst=VQE_args_list[:],
        k=chunks_quantity,
    )
    arguments_process_chunk:List[VQE_process_chunk_args] = list(map(lambda arguments_chunk: VQE_process_chunk_args(arguments_list=arguments_chunk), arguments_chunks))
    # for arguments_chunk in tqdm(arguments_chunks):

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(apply_VQE_to_args_list, arguments_process_chunk)
    
# nohup python /data01/vvkiselev/projects/dpl/MyProject/models/VQE/apply.py > VQE.out & disown
# tail -f VQE.out

# check processed data quantity 
# /home/vvkiselev/data/other/dpl/processed/dusha/crowd
# ls train/wavs/ | wc -l