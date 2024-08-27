from argparse import ArgumentParser
import pytorch_lightning as pl
import torch

def get_args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # Add model args
    parser.add_argument('--base_model_name', type=str, default="bert-base-uncased")
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=128)
    # Add dataset args
    parser.add_argument('--batch_size', type=int, default=32)
    # Add optimizer args
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_epochs', type=int, default=10)
    # Add device-related settings 
    parser.add_argument('--devices', type=int, 
        default=1 if torch.cuda.is_available() else 0) # Use 'devices' for specifying number of GPUs
    parser.add_argument('--accelerator', type=int, 
        default= 'gpu' if torch.cuda.is_available() else 'cpu') # Use 'accelerator' to specify device type
    
    args = parser.parse_args()
    return args