import torch
from pathlib import Path
import pandas as pd
from navec import (
    Navec, 
    vocab,
)
from slovnet.model.emb import NavecEmbedding
from typing import (
    Union,
    List,
)
import sys

sys.path.append('../..')
from configs.paths import (
    DEPLOMA_DIR_PATH,
)

TORCH_UNIFIED_SIZE:int = 49

HYPOTHESES_DIR_PATH:Path = DEPLOMA_DIR_PATH / 'hypotheses'
NAVEC_EMBEDDINGS_FILE_PATH:Path = HYPOTHESES_DIR_PATH / 'navec_hudlit_v1_12B_500K_300d_100q.tar'  # 51MB

NAVEC:Navec = Navec.load(NAVEC_EMBEDDINGS_FILE_PATH)  # ~1 sec, ~100MB RAM
EMB:NavecEmbedding = NavecEmbedding(NAVEC)
VOCAB:vocab.Vocab = NAVEC.vocab

# exapmle, TODO: try to get embeddings size from library var
# words:List[str] = ['навек', '<unk>', '<pad>']
# ids:List = [NAVEC.vocab[_] for _ in words]
# ids_tensor:torch.Tensor = torch.tensor(ids)

# embeddings size: 300 
# NAVEC_EMBEDDINGS_SIZE:int = EMB(ids_tensor).shape[1]
NAVEC_EMBEDDINGS_SIZE:int = EMB(torch.tensor(NAVEC.vocab['навек'])).shape[0]

def word_to_emb(
        word_or_words: Union[str, list[str]], 
        emb:NavecEmbedding=EMB, 
        vocabular:vocab.Vocab=VOCAB, 
        embedding_size:int=NAVEC_EMBEDDINGS_SIZE,
        ) -> torch.Tensor:
    def single_word_to_emb(
        word:str,
        emb:NavecEmbedding = EMB,
        vocabular:vocab.Vocab = VOCAB,
        )->torch.Tensor:
        return emb(torch.tensor(vocabular[word]))
    if type(word_or_words) == list:
        result = []
        for word in word_or_words:
            try:
                word_emb:torch.Tensor = single_word_to_emb(
                    word=word,
                    emb=emb, 
                    vocabular=vocabular,
                )
            except KeyError:
                continue
            result.append(word_emb)
        if len(result) != 0:
            return torch.stack(result)
        else:
            return torch.zeros((1, embedding_size))
    else:
        return single_word_to_emb(
            word=word_or_words,
            emb=emb, 
            vocabular=vocabular, 
        )

def text_2_unified_tensor(
    text: str, 
    torch_unified_size:int = TORCH_UNIFIED_SIZE,
    word_tensor_length:int = NAVEC_EMBEDDINGS_SIZE,
    emb:NavecEmbedding=EMB, 
    vocabular:vocab.Vocab=VOCAB,
    ):
    words_list:List[str] = text.split()
    text_tensor:torch.Tensor = word_to_emb(
        word_or_words=words_list, 
        emb=emb, 
        vocabular=vocabular,
    )

    zero_tensors_quantity:int = max(
        0, 
        torch_unified_size - len(text_tensor)
    )
    zero_tensor:torch.Tensor = torch.zeros((zero_tensors_quantity, word_tensor_length))

    filled_tensor:torch.Tensor = torch.concat([zero_tensor, text_tensor])
    return filled_tensor


def texts_series_2_tensor(
    s:pd.Series,
    torch_unified_size:int = TORCH_UNIFIED_SIZE,
    word_tensor_length:int = NAVEC_EMBEDDINGS_SIZE,
    emb:NavecEmbedding=EMB, 
    vocabular:vocab.Vocab=VOCAB,
    )->torch.Tensor:
    texts_list:List[str] = s.to_list()
    concatted_texts_embeddings:torch.Tensor = torch.stack(
        list(
            map(
                lambda x: text_2_unified_tensor(
                    text=x, 
                    torch_unified_size=torch_unified_size,
                    word_tensor_length=word_tensor_length,
                    emb=emb,
                    vocabular=vocabular,
                ),
                texts_list
            )
        )
    )

    # print('concatted_texts_embeddings.shape', concatted_texts_embeddings.shape)
    return concatted_texts_embeddings

def bag_of_words(
        words: list[str], 
        emb:NavecEmbedding = EMB, 
        vocabular:vocab.Vocab = VOCAB, 
        embedding_size = NAVEC_EMBEDDINGS_SIZE,
    ) -> torch.Tensor:
    word_embeddings:torch.Tensor = word_to_emb(
        emb=emb, 
        vocabular=vocabular, 
        word_or_words=words,
    )
    vector_sum = torch.zeros(embedding_size)
    for word in word_embeddings:
        vector_sum += word
    
    return vector_sum / max(len(word_embeddings), 1)
