import torch
import torch.nn as nn
from pathlib import Path
import librosa
import numpy as np
from dataclasses import (
    dataclass,
)
from enum import Enum
from typing import (
    Optional,
)
import transformers
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from configs.datasets.dusha import (
    SAMPLE_RATE,
)
from configs.paths import (
    W2V_BIO_DIR_PATH,
)

BIO_MODEL_AGE_MULTIPLIER:float = 100.
BIO_MODEL_DEVICE: str = 'cuda:6' if torch.cuda.is_available() else 'cpu'
MODEL_NEEDED_RAM:int = 3 *  (1024 ** 3)

def read_wav_as_array(
    path:Path, 
    sample_rate=SAMPLE_RATE,
    )->np.ndarray:
    # Load audio, force mono, resample if needed
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    # Reshape to (1, N)
    audio = np.expand_dims(audio, axis=0)
    return audio

class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
        ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender
    

class Gender(Enum):
    female:int = 0     
    male:int = 1       
    child:int = 2

@dataclass
class SpeakerBio:
    age: int
    gender: Gender

@dataclass
class Wav2VecBioModelPredProba:
    age:float
    female:float
    male:float
    child:float
    age_multiplier:float = BIO_MODEL_AGE_MULTIPLIER
    
    def to_speaker_bio(
        self,
        )->SpeakerBio:
        return SpeakerBio(
            age=self.age * self.age_multiplier,
            gender=Gender(
                np.array([self.female, self.male, self.child]).argmax(),
            ),
        )

@dataclass
class Wav2VecBioModel:
    model:AgeGenderModel
    processor:Wav2Vec2Processor
    age_multiplier:float = BIO_MODEL_AGE_MULTIPLIER
    @classmethod
    def dir_path_with_device_init(
        cls,
        dir_path:Path = W2V_BIO_DIR_PATH,
        device_name:str = BIO_MODEL_DEVICE,
        age_multiplier:float = BIO_MODEL_AGE_MULTIPLIER,
        ):
        dir_path:str = str(dir_path)
        model:AgeGenderModel = AgeGenderModel.from_pretrained(dir_path).to(device_name)
        processor:Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(dir_path)
        return cls(
            model=model,
            processor=processor,
            age_multiplier=age_multiplier,
        )
    def audio_file_path_2_predict(
        self,
        path:Path,
        sample_rate=SAMPLE_RATE,
        processor_input_valeues_keyword:str = 'input_values',
        )->Optional[Wav2VecBioModelPredProba]:
        if not path.exists():
            print(f'WARNING: audio file with path = {path} does not exist!')
            return None
        signal:np.ndarray = read_wav_as_array(
            path,
            sample_rate=sample_rate,
        )
        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        processor_output:transformers.feature_extraction_utils.BatchFeature = self.processor(
            signal, 
            sampling_rate=sample_rate,
        )

        processor_output = processor_output[processor_input_valeues_keyword][0]
        processor_output = processor_output.reshape(1, -1)
        processor_output = torch.from_numpy(processor_output).to(self.model.device)

        with torch.no_grad():
            model_output = self.model(processor_output)
            model_output = torch.hstack([model_output[1], model_output[2]])

        # convert to numpy
        model_output = model_output.detach().cpu().numpy()
        #    Age        female     male       child
        # [[ 0.33793038 0.2715511  0.2275236  0.5009253 ]]
        row:np.ndarray = model_output[0]
        del processor_output, model_output
        torch.cuda.empty_cache()
        return Wav2VecBioModelPredProba(age=row[0], female=row[1], male=row[2], child=row[3], age_multiplier=self.age_multiplier)
