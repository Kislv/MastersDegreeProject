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
from typing import (
    List,
)
import sys
from dataclasses import dataclass
sys.path.append('../..')
from configs.paths import (
    # WHISPER_MEDIUM_DIR_PATH,
    WHISPER_LARGE_DIR_PATH as WHISPER_DIR_PATH,
    PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_DIR_PATH,
)
from configs.base import (
    DOT_CSV,
)
from models.config import (
    TORCH_TENSORS_KEYWOED,
    MOST_EFFECTIVE_AVAILABLE_DEVICE,
)

# load model and processor
# WHISPER_MODEL_NAME:str = 'openai/whisper-medium'
WHISPER_LARGE_V3_MODEL_NAME:str = 'openai/whisper-large-v3'
WHISPER_LARGE_MODEL_NEEDED_RAM:int = 27 *  (1024 ** 3)
# WHISPER_PROCESSOR = WhisperProcessor.from_pretrained(WHISPER_LARGE_V3_MODEL_NAME)
# WHISPER_MODEL = WhisperForConditionalGeneration.from_pretrained(WHISPER_LARGE_V3_MODEL_NAME)
WHISPER_PROCESSOR = None
WHISPER_MODEL = None
# WHISPER_MODEL.config.forced_decoder_ids = None
PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_WHISPER_LARGE_V3_FILE_PATH:Path = PROCESSED_DUSHA_CROWD_TRANSCRIPTIONS_DIR_PATH / (Path(WHISPER_LARGE_V3_MODEL_NAME).name + DOT_CSV)

# Define model name and local directory
# WHISPER_MEDIUM_DIR_PATH:Path = Path('/data01/vvkiselev/data/dpl/models/whisper_medium')

# Download model files to local directory if not already downloaded
if not os.path.exists(WHISPER_DIR_PATH):
    snapshot_download(
        repo_id=WHISPER_LARGE_V3_MODEL_NAME, 
        local_dir=WHISPER_DIR_PATH,
    )

# Load model and processor from local directory
# WHISPER_MODEL:transformers.models.whisper.modeling_whisper.WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(WHISPER_DIR_PATH)
# WHISPER_PROCESSOR:transformers.models.whisper.processing_whisper.WhisperProcessor = WhisperProcessor.from_pretrained(WHISPER_DIR_PATH)
# WHISPER_MODEL = WhisperForConditionalGeneration.from_pretrained(WHISPER_DIR_PATH).to(MOST_EFFECTIVE_AVAILABLE_DEVICE)
# WHISPER_MODEL.config.forced_decoder_ids = None

@dataclass
class Whisper:
    processor:WhisperProcessor
    model:WhisperForConditionalGeneration
    @classmethod
    def device_name_dir_path_init(
        cls,
        device_name:str = MOST_EFFECTIVE_AVAILABLE_DEVICE,
        whisper_dir_path:Path = WHISPER_DIR_PATH,
        ):
        processor:WhisperProcessor = WhisperProcessor.from_pretrained(whisper_dir_path)
        model:WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(whisper_dir_path).to(device_name)
        model.config.forced_decoder_ids = None
        return cls(
            processor=processor,
            model=model,
        )

    def _tensor_with_sr_transcription(
        self,
        tensor: torch.Tensor,
        sr: int,
        ) -> List[str]:
        tensor:np.ndarray = tensor.numpy().squeeze()
        inputs = self.processor(
            tensor,
            sampling_rate=sr,
            return_tensors=TORCH_TENSORS_KEYWOED,
            language="russian",  
            task="transcribe",
        )
        input_features = inputs.input_features
        attention_mask = getattr(inputs, "attention_mask", None)

        device:torch.device = next(self.model.parameters()).device

        input_features = input_features.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            if attention_mask is not None:
                predicted_ids = self.model.generate(input_features, attention_mask=attention_mask)
            else:
                predicted_ids = self.model.generate(input_features)
            transcription:List[str] = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        del input_features, attention_mask
        torch.cuda.empty_cache()
        return transcription

    def audio_file_2_transcription(
        self,
        audio_path:Path,
        ) -> List[str]:
        tensor, sr = torchaudio.load(audio_path)
        return self._tensor_with_sr_transcription(
            tensor=tensor,
            sr=sr,
        )
    
