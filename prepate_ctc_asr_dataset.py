import json
import os
import torchaudio
from torch.utils.data import Dataset, DataLoader

import numpy as np

import torch
import torchaudio.transforms as T

import nemo.collections.asr as nemo_asr



class AudioAugmentation:
    def __init__(self, sample_rate=48000, quantization_channels=128, snr=25):
        self.transforms_mulaw = torch.nn.Sequential(
            T.MuLawEncoding(quantization_channels = quantization_channels),
            T.MuLawDecoding(quantization_channels = quantization_channels),
        )
        self.add_noise = T.AddNoise()
        self.snr = snr

    def __call__(self, waveform):
        waveform = self.transforms_mulaw(waveform)
        noises = torch.rand_like(waveform)
        waveform = self.add_noise(waveform, noises, torch.tensor([[self.snr]]))
        return waveform

class AudioDataset(Dataset):
    def __init__(self, manifest_path, audio_dir, transform=None, file_format='.opus'):
        self.audio_dir = audio_dir
        self.transform = transform
        
        self.file_format = file_format
        
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
        audio_path = f"{self.audio_dir}/{item['audio_filepath']}"
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Apply any transformations if provided
        if self.transform:
            waveform = self.transform(waveform)
        
        # Return the audio, text, and any other metadata you want
        return {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'text': item['text'],
            'duration': item['duration'],
            'audio_path': audio_path
        }

if __name__ == "__main__":
    # Path to your manifest and audio directory
    manifest_path = '/data/golos/10min.jsonl'
    output_file_path = '/data/golos/10min_logprobs.jsonl'
    
    # manifest_path = '/data/golos/1hour.jsonl'
    # output_file_path = '/data/golos/1hour_logprobs.jsonl'
    
    # manifest_path = '/data/golos/10hours.jsonl'
    # output_file_path = '/data/golos/10hours_logprobs.jsonl'
        
    # manifest_path = '/data/golos/100hours.jsonl'
    # output_file_path = '/data/golos/100hours_logprobs.jsonl'
    
    audio_dir = '/data/golos/train_opus'

    # manifest_path = '/data/golos/test_opus/farfield/farfield.jsonl'
    # output_file_path = '/data/golos/test_farfield_logprobs.jsonl'
    # audio_dir = '/data/golos/test_opus/farfield/'
    
    # CTC-model and other settings
    model_name="stt_ru_conformer_ctc_large"
    audio_file_ext = '.opus'
    logit_file_ext = '.npy'
    batch_size = 1
    max_length = 1024
    output_logprobs = True
    
    # Create the dataset
    dataset = AudioDataset(manifest_path, audio_dir, file_format=audio_file_ext)
    audiofiles = [os.path.join(audio_dir, _['audio_filepath']) for _ in dataset.manifest]
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Augment, save and recognize files
    aug_audio_files = []
    augmentor = AudioAugmentation()
    for i, batch in enumerate(dataloader):
        waveform = batch['waveform']
        audio_path = batch['audio_path'][-1]
        waveform = augmentor(torch.tensor(waveform))
        aug_audio_path = audio_path.replace(audio_file_ext,'_aug'+audio_file_ext)
        torchaudio.save(aug_audio_path, waveform.squeeze(0), batch['sample_rate'])
        aug_audio_files.append(aug_audio_path)
        continue
    
    # Get CTC-logits from a dataset  
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
    log_probs = asr_model.transcribe(audiofiles, batch_size = 16, logprobs=True)
    
    aug_log_probs = asr_model.transcribe(aug_audio_files, batch_size = 16, logprobs=True)
    
    # Iterate through the dataloader and make new manifest
    new_manifest = []
    for i, batch in enumerate(dataloader):
        print(batch['waveform'].shape)
        print(batch['sample_rate'])
        print(batch['text'])
        print(batch['duration'])
        print(batch['audio_path'])
        
        new_sample = {}
        
        sample_rate = batch['sample_rate']
        new_sample['sample_rate'] = sample_rate.item()
        text = batch['text']
        new_sample['text'] = text[0]
        duration = batch['duration']
        new_sample['duration'] = duration.item()
        
        audio_path = batch['audio_path'][0]
        
        audio_subpath = dataset.manifest[i]['audio_filepath']
        
        if not audio_subpath in audio_path:
            raise
        
        new_sample['audio_filepath'] = audio_subpath
               
        log_prob = log_probs[i]
        aug_log_prob = aug_log_probs[i]
        
        if log_prob.shape[0] > max_length: continue
        
        np_subpath = audio_subpath.replace(audio_file_ext, logit_file_ext)
        np.save(os.path.join(audio_dir, np_subpath), log_prob)
        
        new_sample['logits_filepath'] = np_subpath
        new_manifest.append(new_sample)
        
        np_subpath = audio_subpath.replace(audio_file_ext, '_aug'+logit_file_ext)
        np.save(os.path.join(audio_dir, np_subpath), aug_log_prob)
        
        new_sample_aug = new_sample.copy()
        new_sample_aug['logits_filepath'] = np_subpath
        new_manifest.append(new_sample_aug)        
        continue

    # Write the list of dictionaries to a .jsonl file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for entry in new_manifest:
            json.dump(entry, file)
            file.write('\n') 

    #TODO use dataset to create loggits dataset

    print('done!')