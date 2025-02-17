import torch
import torchaudio
import numpy as np
from pathlib import Path

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# load model and processor
whisper_model_name:str = 'openai/whisper-medium'
WHISPER_PROCESSOR = WhisperProcessor.from_pretrained(whisper_model_name)
WHISPER_MODEL = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
WHISPER_MODEL.config.forced_decoder_ids = None
TORCH_TENSORS_KEYWOED:str = 'pt'

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
