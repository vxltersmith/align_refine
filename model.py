import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertModel
from nemo.collections.asr.losses import CTCLoss
import hydra
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_greedy_decoding import GreedyCTCInfer
from nemo.collections.asr.metrics.wer import word_error_rate
import math


def get_sinusoidal_embedding(step_index, hidden_dim):
    # Initialize the positional encoding matrix
    position = torch.tensor([step_index], dtype=torch.float).unsqueeze(1)  # Shape [1, 1]
    div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))  # Shape [hidden_dim / 2]

    # Calculate the positional encoding
    sinusoidal_embedding = torch.zeros(hidden_dim)
    sinusoidal_embedding[0::2] = torch.sin(position * div_term)
    sinusoidal_embedding[1::2] = torch.cos(position * div_term)

    return sinusoidal_embedding.view(1, 1, hidden_dim)  # Reshape to [1, hidden_dim, 1]

class CTCBertModel(pl.LightningModule):
    base_model_name: str
    vocab_size: int
    
    def __init__(self, base_model_name, vocab_size, 
            refine_n_steps = 1, dropout = 0.001, transforms_cfg = None, mix_wth_factor=0.5,
            transform_eval = False, new_max_position_embeddings = None, 
            extend_positional_embeddings_on_load = False):
        super(CTCBertModel, self).__init__()
        
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size
        self.ctc_outputs_size = vocab_size+1 # including blank symbol
        self.transforms = transforms_cfg
        self.refine_n_steps = refine_n_steps
        self.dropout = dropout
        self.checkpoint = None
        self.mix_wth_factor = mix_wth_factor
        self.transform_eval = transform_eval
        self.new_max_position_embeddings = new_max_position_embeddings
        self.extend_positional_embeddings_on_load = extend_positional_embeddings_on_load
        
    def setup(self, stage=None):       
        self.model = BertModel.from_pretrained(self.base_model_name)
        
        model_config = self.model.config       
        self.input_embedding_layer = nn.Linear(self.ctc_outputs_size, model_config.hidden_size)
        self.output_embedding_layer = nn.Linear(model_config.hidden_size, self.ctc_outputs_size)
                
        if self.checkpoint is not None:
            if self.extend_positional_embeddings_on_load: 
                self.do_extend_positional_embeddings()
            self.load_state_dict(self.checkpoint['state_dict'])
        else:
            # Initialize weights for custom layers
            nn.init.xavier_uniform_(self.input_embedding_layer.weight)
            nn.init.xavier_uniform_(self.output_embedding_layer.weight)
            
        if (self.new_max_position_embeddings is not None and 
            self.model.config.max_position_embeddings != self.new_max_position_embeddings):
            self.do_extend_positional_embeddings()
            
        self.__set_dropout__()
        
        if self.transforms is not None:
            transforms_sequential = torch.nn.Sequential()
            for _ in self.transforms: transforms_sequential.append(_)
            self.transforms = transforms_sequential
        
        self.loss_fn = CTCLoss(self.vocab_size, zero_infinity=True)
        
    def do_extend_positional_embeddings(self):
        new_max_length = self.new_max_position_embeddings
        model = self.model
        model.embeddings.token_type_ids = torch.zeros(1, new_max_length).long()
        # Get the original position embeddings
        old_position_embeddings = model.embeddings.position_embeddings
        # Create a new position embedding layer with the new max length
        new_position_embeddings = nn.Embedding(new_max_length, old_position_embeddings.embedding_dim)
        # Copy the weights from the original position embeddings to the new layer
        new_position_embeddings.weight.data[:old_position_embeddings.weight.size(0)] = old_position_embeddings.weight.data
        # Replace the model's position embeddings with the new layer
        model.embeddings.position_embeddings = new_position_embeddings
        model.embeddings.position_ids = torch.arange(1, new_max_length).unsqueeze(0)
        # Update the model's configuration
        model.config.max_position_embeddings = new_max_length
        
    def set_checkpoint(self, full_path: str):
        # Load the state dict from the checkpoint
        self.checkpoint = torch.load(full_path, map_location=self.device)  # or 'cuda' for GPU
        
    def __set_dropout__(self):
        model = self.model
        # Disable dropout by setting the dropout probability to 0
        dropout_value = self.dropout
        model.config.hidden_dropout_prob = dropout_value
        model.config.attention_probs_dropout_prob = dropout_value
        # Manually update the dropout layers in the model
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_value

    def forward(self, softmaxed_input, input_lengths, original_input = None, i = 0):
        # Project logits into Bert model embedding space
        embedded_input = self.input_embedding_layer(softmaxed_input)
        embedded_input = embedded_input + get_sinusoidal_embedding(i, embedded_input.shape[-1]).to(embedded_input.device)
        
        if original_input is not None:
            original_embedded_input = self.input_embedding_layer(original_input)
            embedded_input = (embedded_input + self.mix_wth_factor * original_embedded_input)
        
        # Create attention mask
        real_max_length = input_lengths.max()
        batch_size = input_lengths.shape[0]
        attention_mask = (torch.arange(real_max_length).expand(batch_size, real_max_length).to(self.device) < input_lengths.unsqueeze(1)).long()
        
        # Infer with Bert
        outputs = self.model(inputs_embeds=embedded_input, attention_mask=attention_mask)
        outputs = outputs.last_hidden_state
        
        # Project Bert embeddigs back to original space and convert into logits
        outputs = self.output_embedding_layer(outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        
        return outputs

    def training_step(self, batch, batch_idx):
        input_logprobs, labels_batch, input_lengths, label_lengths = batch
        
        if self.transforms is not None:
            input_logprobs = self.transforms(input_logprobs)

        ctc_outputs = input_logprobs
        loss = 0
        for _ in range(self.refine_n_steps):               
            # Forward pass
            ctc_outputs = self(ctc_outputs, input_lengths, input_logprobs, _)
            # Calculate loss
            loss += self.loss_fn(
                log_probs=ctc_outputs,
                targets=labels_batch,
                input_lengths=input_lengths,
                target_lengths=label_lengths
            )
            continue
        
        # Log the loss
        self.log('train_loss', loss/self.refine_n_steps, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_logprobs, labels_batch, input_lengths, label_lengths = batch
        
        if self.transforms is not None and self.transform_eval:
            input_logprobs = self.transforms(input_logprobs)
        
        ctc_outputs = input_logprobs
        loss = 0
        refined_list = []
        wers = []
        for _ in range(self.refine_n_steps):           
            # Forward pass
            ctc_outputs = self(ctc_outputs, input_lengths, input_logprobs, _)
            refined_list.append(ctc_outputs)
            # Calculate loss
            loss += self.loss_fn(
                log_probs=ctc_outputs,
                targets=labels_batch,
                input_lengths=input_lengths,
                target_lengths=label_lengths
            )/self.refine_n_steps
            continue

        self.print_predictions(refined_list, input_logprobs, input_lengths, labels_batch, label_lengths, False)

        # Log the loss
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def print_predictions(self, predictions, logits_input, input_lengths, labels_batch, label_lengths, random=True):
        batchid_to_show = torch.randint(0, labels_batch.shape[0], (1, 1)).item()*int(random)# show random predictions from batch
        
        # decode gt labels
        target_ids = labels_batch.long().cpu().numpy().tolist()
        tokenizer = self.ctc_tokenizer.tokenizer
        gt_texts = [tokenizer.decode_ids(target_ids[_][:label_lengths[_]]) for _ in range(len(target_ids))]
        gt_text = gt_texts[batchid_to_show]
        
        # decode input logits
        decoded_batch = self.greedy_decoder.forward(decoder_output = logits_input, decoder_lengths = input_lengths)
        decoded_batch = decoded_batch[0]
        decoded_batch = self.bpe_decoder.decode_hypothesis(decoded_batch, fold_consecutive=True)
        decoded_input_batch = [_.words for _ in decoded_batch]
        input_text = " ".join(decoded_input_batch[batchid_to_show])
        
        predicted_texts = []
        if not isinstance(predictions, list): predictions = [predictions]
        # decode predicted logits
        for i, predictions_tensor in enumerate(predictions):
            decoded_batch = self.greedy_decoder.forward(decoder_output = predictions_tensor, decoder_lengths = input_lengths)
            decoded_batch = decoded_batch[0]
            decoded_batch = self.bpe_decoder.decode_hypothesis(decoded_batch, fold_consecutive=True)
            decoded_predicted_texts = [_.words for _ in decoded_batch]
            
            wer_origs = word_error_rate(decoded_predicted_texts, decoded_input_batch, True)
            self.log(f'validation_wer_origs_refine_step_{i}', wer_origs, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, sync_dist=True)

            wer_origs = word_error_rate(decoded_predicted_texts, gt_texts, True)
            self.log(f'validation_wer_refine_step_{i}', wer_origs, on_step=False, on_epoch=True, 
                    prog_bar=False, logger=True, sync_dist=True)
            
            predicted_text = " ".join(decoded_predicted_texts[batchid_to_show])
            predicted_texts.append(predicted_text)
        
        self.print('\n tartget: ' + gt_text)
        self.print('\n input: ' + input_text)
        for _ in range(len(predicted_texts)):
            self.print(f'\n refine_step[{_}]: ' + predicted_texts[_])
        return
    
    def set_ctc_tokenizer(self, ctc_tokenizer):
        """
        Set tokenizer to a model 
        """
        self.ctc_tokenizer = ctc_tokenizer
        config = CTCBPEDecodingConfig(
            preserve_alignments=False,
            compute_timestamps=False,
        )
        self.greedy_decoder = GreedyCTCInfer(self.vocab_size)
        self.bpe_decoder = CTCBPEDecoding(config, ctc_tokenizer)
    
    def set_optimizer_configs(self, optimizer_cfg, lr_scheduler_cfg = None):
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg

    def configure_optimizers(self):
        # create optimizer from configuration composed by Hydra
        parameters = list(self.parameters())
        print(parameters)
        optim = hydra.utils.instantiate(self.optimizer_cfg, params=parameters)
        if not self.lr_scheduler_cfg:
            return [optim]
        # Create scheduler from configuration compose by Hydra if True selected in use_lr_scheduler config
        scheduler = hydra.utils.instantiate(self.lr_scheduler_cfg, optimizer=optim)
        return [optim], [scheduler]

    def collate_fn(self, batch):
        input_batch, labels_batch, input_lengths, label_lengths = zip(*batch)
        input_batch = torch.stack(input_batch)
        labels_batch = torch.stack(labels_batch)
        input_lengths = torch.tensor(input_lengths)
        label_lengths = torch.tensor(label_lengths)
        return input_batch, labels_batch, input_lengths, label_lengths

if __name__ == '__main__':
    # Example parameters
    model_name = "bert-base-uncased"
    batch_size = 32
    max_input_length = 256
    vocab_size = 128
    ctc_outputs_size = vocab_size + 1

    # Create model
    model = CTCBertModel(model_name, vocab_size, ctc_outputs_size, max_input_length)
