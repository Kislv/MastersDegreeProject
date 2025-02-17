from pathlib import Path

DEPLOMA_DIR_PATH = Path('/home/vkiselev/data/other/univer/deploma')
DUSHA_DIR_PATH = DEPLOMA_DIR_PATH / 'dusha'
DUSHA_CROWD_DIR_PATH = DUSHA_DIR_PATH / 'crowd'
DUSHA_CROWD_TRAIN_DIR_PATH = DUSHA_CROWD_DIR_PATH / 'crowd_train/'
DUSHA_CROWD_TRAIN_FILE_PATH = DUSHA_CROWD_TRAIN_DIR_PATH/ 'raw_crowd_train.tsv'
DUSHA_CROWD_TEST_DIR_PATH = DUSHA_CROWD_DIR_PATH / 'crowd_test/'
RAW_CROWD_TEST_PATH = DUSHA_CROWD_TEST_DIR_PATH / 'raw_crowd_test.tsv'

def crowd_test_wav_path_2_abs_path(
    path:str,
    crowd_test_path:Path = DUSHA_CROWD_TEST_DIR_PATH,
    )->Path:
    return crowd_test_path  / path

