import torch
import torchaudio
import numpy as np
from pathlib import Path
import os
import transformers
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from huggingface_hub import snapshot_download
import sys

sys.path.append('../..')
from configs.paths import (
    WHISPER_MEDIUM_DIR_PATH,
)
from models.config import (
    TORCH_TENSORS_KEYWOED,
)


# load model and processor
WHISPER_MODEL_NAME:str = 'openai/whisper-medium'
# WHISPER_PROCESSOR = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
# WHISPER_MODEL = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME)
WHISPER_PROCESSOR = None
WHISPER_MODEL = None
# WHISPER_MODEL.config.forced_decoder_ids = None


# Define model name and local directory
# WHISPER_MEDIUM_DIR_PATH:Path = Path('/data01/vvkiselev/data/dpl/models/whisper_medium')

# Download model files to local directory if not already downloaded
if not os.path.exists(WHISPER_MEDIUM_DIR_PATH):
    snapshot_download(
        repo_id=WHISPER_MODEL_NAME, 
        local_dir=WHISPER_MEDIUM_DIR_PATH,
    )

# Load model and processor from local directory
WHISPER_PROCESSOR:transformers.models.whisper.processing_whisper.WhisperProcessor = WhisperProcessor.from_pretrained(WHISPER_MEDIUM_DIR_PATH)
WHISPER_MODEL:transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(WHISPER_MEDIUM_DIR_PATH)
WHISPER_MODEL.config.forced_decoder_ids = None

def whisper_tensor_with_sr_transcription(
    tensor:torch.Tensor,
    sr:int,
    processor:WhisperProcessor=WHISPER_PROCESSOR,
    model:WhisperForConditionalGeneration=WHISPER_MODEL,
    )->str:
    tensor:np.ndarray = tensor.numpy().squeeze()
    input_features:torch.Tensor = processor(tensor, sampling_rate=sr, return_tensors=TORCH_TENSORS_KEYWOED).input_features 

    # generate token ids
    predicted_ids:torch.Tensor = model.generate(input_features)
    # decode token ids to text
    transcription:str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

def whisper_audio_file_2_transcription(
    audio_path:Path,
    processor:WhisperProcessor=WHISPER_PROCESSOR,
    model:WhisperForConditionalGeneration=WHISPER_MODEL,
    ):
    tensor, sr = torchaudio.load(audio_path)
    return whisper_tensor_with_sr_transcription(
        tensor=tensor,
        sr=sr,
        processor=processor,
        model=model,
    )
