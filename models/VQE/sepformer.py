from dataclasses import dataclass
from pathlib import Path
from speechbrain.inference import SepformerSeparation as separator
import torchaudio
import torch
import wave


from configs.paths import (
    SEPFORMER_16KHZ_DIR_PATH,
)
from configs.base import (
    RB_OPEN_FILE_MODE,
)

from models.config import (
    MOST_EFFECTIVE_AVAILABLE_DEVICE,
)

SEPFORMER_DEVICE_FIELD_NAME:str = 'device'
@dataclass
class Sepformer:
    _model:separator
    @classmethod
    def file_init(
        cls,
        path:Path = SEPFORMER_16KHZ_DIR_PATH,
        device:torch.device = MOST_EFFECTIVE_AVAILABLE_DEVICE,
        device_field_name:str = SEPFORMER_DEVICE_FIELD_NAME,
        ):
        path_str = str(path)
        # device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        return cls(
            _model = separator.from_hparams(
                source=path_str,
                savedir=path_str,
                run_opts={device_field_name: device},
            ),
        )
    def apply_VQE(
        self,
        input_file_path:Path,
        output_file_path:Path,
        open_file_mode:str=RB_OPEN_FILE_MODE,
        )->None:
        # Perform speech enhancement (separation)
        est_sources:torch.Tensor = self._model.separate_file(path=str(input_file_path))
        # est_sources shape: (batch, channels, samples)
        # The first channel (index 0) is usually the enhanced speech
        enhanced_audio:torch.Tensor = est_sources[:, :, 0].detach().cpu()

        with wave.open(str(input_file_path), open_file_mode) as wav_file:
            sr:int = wav_file.getframerate()
        # Save the enhanced audio to a new WAV file with 16 kHz sampling rate
        torchaudio.save(output_file_path, enhanced_audio, sr) # SR from readed file

