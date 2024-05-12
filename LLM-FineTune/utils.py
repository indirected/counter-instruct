import datasets
from omegaconf import DictConfig
import hydra
import torch

def prepare_data(datacfg: DictConfig) -> datasets.DatasetDict:
    dataset = hydra.utils.instantiate(datacfg.dataset.HF_cls)
    train_size = int(len(dataset)*datacfg.split_sizes.train)
    val_size = int(len(dataset)*datacfg.split_sizes.val)
    test_size = len(dataset) - train_size - val_size

    train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(datacfg.split_seed))

    dataset = datasets.DatasetDict({
        'train': dataset.select(train.indices),
        'val': dataset.select(val.indices),
        'test': dataset.select(test.indices)
    })

    return dataset

def prepare_model_and_tokenizer(modelcfg: DictConfig) -> tuple:
    model = hydra.utils.instantiate(modelcfg.model_HF_cls)
    tokenizer = hydra.utils.instantiate(modelcfg.tokenizer_HF_cls)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    return model, tokenizer