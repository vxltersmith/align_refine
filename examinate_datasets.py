import json
import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import nemo.collections.asr as nemo_asr
import numpy as np
from prepate_ctc_asr_dataset import AudioDataset

if __name__ == "__main__":
    batch_size = 1
    audio_file_ext = '.opus'

    # Path to your manifest and audio directory      
    audio_dir = '/data/golos/train_opus'    
    manifest_path = '/data/golos/10min_logprobs.jsonl'
    output_file_path = '/data/golos/10min_logprobs.jsonl'
    
    # Create first dataset
    first_dataset = AudioDataset(manifest_path, audio_dir, file_format=audio_file_ext)
    first_dataset_audiofiles = [os.path.join(audio_dir, _['audio_filepath']) for _ in first_dataset.manifest]
    # Create the dataloader
    first_dataset_dataloader = DataLoader(first_dataset, batch_size=batch_size, shuffle=False)
    
    # Create second dataset
    manifest_path = '/data/golos/10hours.jsonl'
    output_file_path = '/data/golos/10hours_logprobs.jsonl'
    second_dataset = AudioDataset(manifest_path, audio_dir, file_format=audio_file_ext)
    second_dataset_audiofiles = [os.path.join(audio_dir, _['audio_filepath']) for _ in second_dataset.manifest]
    # Create the dataloader
    second_dataset_dataloader = DataLoader(second_dataset, batch_size=batch_size, shuffle=False)
    
    common_paths  =[]
    for path in first_dataset_audiofiles:
        if path in second_dataset_audiofiles:
            common_paths.append(path)
        continue
    print('done!')