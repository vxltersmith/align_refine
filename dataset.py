import torch
import pytorch_lightning as pl
from data_utils import generate_batch
import json
import os
from torch.utils.data import Dataset
import numpy as np
from nemo.collections.common.tokenizers import SentencePieceTokenizer


class RandomCTCDataset(torch.utils.data.Dataset):
    def __init__(self, max_input_length, vocab_size):
        self.max_input_length = max_input_length
        self.vocab_size = vocab_size
        
        self.data = {}

    def __len__(self):
        return 1000  # Change this to the actual size of your dataset

    def __getitem__(self, idx):
        if not idx in self.data.keys():
            sample = generate_batch(1, self.max_input_length, self.vocab_size)           
            self.data[idx] = sample
            return sample
        return self.data[idx]
    
class LogitCTCDataset(Dataset):
    def __init__(self, manifest_path, audio_dir, tokenizer, file_format='.opus', keep_in_memory=False):
        self.audio_dir = audio_dir
       
        self.file_format = file_format
        
        self.tokenizer = tokenizer
        
        self.keep_in_memory = keep_in_memory
        
        self.logits_cache = {}
        
        # Load the manifest file
        self.manifest = []
        with open(manifest_path, 'r') as file:
            for line in file:
                record = json.loads(line)
                
                audio_path = record['audio_filepath']
                
                if not audio_path.endswith(self.file_format):
                    audio_path = audio_path.replace('.wav', self.file_format)
                    
                record['audio_filepath'] = audio_path
                
                self.manifest.append(record)
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        # Get the metadata for the specific index
        item = self.manifest[idx]
        
        # Construct the full path to the audio file
        np_path = os.path.join(self.audio_dir, item['logits_filepath'])
        
        # Load the audio file and get its lenght
        if self.keep_in_memory and idx in self.logits_cache:
            logits = self.logits_cache[idx]
        else:
            logits = np.load(np_path)
            self.logits_cache[idx] = logits
            
        logits = torch.tensor(logits)
        
        logits_length = torch.tensor(logits.shape[0])
        
        logits = logits.unsqueeze(0)
        
        # Get annotation, tokenize it and get its lenght
        text = item['text']
        tokenized_text = self.tokenizer.tokenizer.Encode(text)
        tokenized_text = torch.tensor(tokenized_text)
        
        label_lengths = torch.tensor(len(tokenized_text))
        
        # Return the audio, text, and any other metadata you want
        return logits, tokenized_text, logits_length, label_lengths

def random_dataset_collate_fn(batch):
    input_batch, labels_batch, input_lengths, label_lengths = zip(*batch)
    
    max_input_length = max([x.shape[1] for x in input_batch])
    max_label_length = max([x.shape[1] for x in labels_batch])
    
    input_batch_stacked = torch.zeros(len(input_batch), max_input_length, input_batch[0].shape[-1])
    for i, _ in enumerate(input_batch):  input_batch_stacked[i, :_.shape[1], :] = _
    labels_batch_stacked = torch.zeros(len(labels_batch), max_label_length)
    for i, _ in enumerate(labels_batch):  labels_batch_stacked[i, :_.shape[1]] = _   
        
    input_batch = input_batch_stacked
    labels_batch = labels_batch_stacked
    input_lengths = torch.tensor(input_lengths)
    label_lengths = torch.tensor(label_lengths)
    return input_batch, labels_batch, input_lengths, label_lengths

def logprobs_dataset_collate_fn(batch):
    input_batch, labels_batch, input_lengths, label_lengths = zip(*batch)
    
    input_lengths = torch.tensor(input_lengths)
    label_lengths = torch.tensor(label_lengths)
    
    max_input_length = input_lengths.max().item()
    max_label_length = label_lengths.max().item()
    
    input_batch_stacked = torch.zeros(len(input_batch), max_input_length, input_batch[0].shape[-1])
    for i, _ in enumerate(input_batch):  input_batch_stacked[i, :_.shape[1], :] = _
    labels_batch_stacked = torch.zeros(len(labels_batch), max_label_length)

    for i, _ in enumerate(labels_batch):  labels_batch_stacked[i, :_.shape[-1]] = _   
        
    input_batch = input_batch_stacked
    labels_batch = labels_batch_stacked

    return input_batch, labels_batch, input_lengths, label_lengths

class CTCDataModule(pl.LightningDataModule):
    max_input_length: int
    vocab_size: int
    
    def __init__(self, batch_size, tokenizer_path, dataset_config, val_dataset_config, num_workers=0, dataset_type=None, val_batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.num_workers = num_workers
        # Set dataset configs
        dataset_type = dataset_type if dataset_type is not None else DatasetTypes.Random
        self.dataset_type = dataset_type
        self.dataset_config = dataset_config
        self.val_dataset_config = val_dataset_config
        # Build tokenizer
        self.tokenizer = SentencePieceTokenizer(tokenizer_path)
        self.tokenizer.tokenizer.vocab_size = self.tokenizer.tokenizer.vocab_size()
        self.collate_fn = random_dataset_collate_fn if self.dataset_type == DatasetTypes.Random else logprobs_dataset_collate_fn
        return

    def setup(self, stage=None):
        dataset_config = self.dataset_config
        val_dataset_config = self.val_dataset_config
        
        if self.dataset_type == DatasetTypes.Random:
            self.dataset = RandomCTCDataset(dataset_config.max_input_length, dataset_config.vocab_size)
            self.val_dataset = RandomCTCDataset(dataset_config.max_input_length, dataset_config.vocab_size)
            
        elif self.dataset_type == DatasetTypes.PreComputeASR:
            self.dataset = LogitCTCDataset(
                dataset_config['manifest_path'], 
                dataset_config['audio_dir'], 
                file_format=dataset_config['file_format'],
                tokenizer=self.tokenizer
            )
            self.val_dataset = LogitCTCDataset(
                val_dataset_config['manifest_path'], 
                val_dataset_config['audio_dir'], 
                file_format=val_dataset_config['file_format'],
                tokenizer=self.tokenizer
            )
        else:
            raise

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            collate_fn= self.collate_fn,
            num_workers=self.num_workers,
            shuffle=True
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size,
            collate_fn= self.collate_fn,
            num_workers=self.num_workers
        )
    
    def get_ctc_tokenizer(self):
        return self.tokenizer

class DatasetTypes:
    Random = 'random'
    PreComputeASR = 'precompute_asr'
        
if __name__ == '__main__':
    batch_size = 32
    max_input_length = 256
    vocab_size = 128
    ctc_outputs_size = vocab_size + 1

    # Create data module
    data_module = CTCDataModule(batch_size, max_input_length, vocab_size)

    # Print shapes and lengths for verification
    for batch in data_module.train_dataloader():
        input_batch, labels_batch, input_lengths, label_lengths = batch
        print(f'Input batch shape: {input_batch.shape}')
        print(f'Labels batch shape: {labels_batch.shape}')
        print(f'Input lengths: {input_lengths}')
        print(f'Label lengths: {label_lengths}')
        break  # Remove this break to iterate over entire dataloader
