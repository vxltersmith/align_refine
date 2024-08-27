import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
import os
import torch


def read_audio_dataset_folder(root_dir_path):
    dir_contents = os.listdir(root_dir_path)
    aduio_text = [
        (os.path.join(root_dir_path, _), 
         os.path.join(root_dir_path, _.replace('.txt', '.opus'))) 
        for _ in dir_contents 
        if _.endswith('.txt') and _.replace('.txt', '.opus') in dir_contents
    ]
    
    batch = []
    for (text_path, audio_path) in aduio_text:
        # Open the file in read mode ('r')
        with open(text_path, 'r') as file:
            # Read the contents of the file
            content = file.read()
            # Print the contents
            print(content)
        
        batch.append((content, audio_path))
    return batch


def main(root_dir_path = '/data/asr_public_phone_calls_2/0/00/', 
        model_name="stt_ru_conformer_ctc_large"):
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
    
    batch = read_audio_dataset_folder(root_dir_path)
    batch_dict = {new: original for original, new in batch}

    audio_paths = [_[-1] for _ in batch]
    targets = [batch_dict[_] for _ in audio_paths]
        
    log_probs = asr_model.transcribe(audio_paths, batch_size = 16, logprobs=True)

    argmaxed = torch.argmax(torch.tensor(log_probs[0]), dim=-1, keepdim=True)
    
    wer = word_error_rate(recognized_results, targets)
    print(wer)

    
if __name__ == '__main__':
    main()
    print('done!')
