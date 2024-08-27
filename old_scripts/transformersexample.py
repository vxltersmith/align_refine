import torch
import torch.nn as nn
from transformers import BertModel
from nemo.collections.asr.losses import CTCLoss
from data_utils import generate_batch


# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
# Example params
batch_size = 32
max_input_length = 256
vocab_size = 128
ctc_outputs_size = vocab_size + 1

input_batch, labels_batch, input_lengths, label_lengths = generate_batch(batch_size, max_input_length, vocab_size)

print(f'Input batch shape: {input_batch.shape}')
print(f'Labels batch shape: {labels_batch.shape}')
print(f'Input lengths: {input_lengths}')
print(f'Label lengths: {label_lengths}')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example softmaxed input (batch_size, max_sequence_length, vocab_size)
softmaxed_input = nn.functional.log_softmax(
    input_batch, dim=-1
).to(device)  # Random softmaxed input for illustration

# Mask to indicate valid token positions
real_max_length = input_lengths.max()
attention_mask = (torch.arange(real_max_length).expand(batch_size, real_max_length) < max_input_length).unsqueeze(1)
attention_mask = attention_mask.to(device).long()

# Initialize BERT model and embedding layers
model = BertModel.from_pretrained(model_name).to(device)
input_embedding_layer = nn.Linear(ctc_outputs_size, model.config.hidden_size).to(device)
output_embedding_layer = nn.Linear(model.config.hidden_size, ctc_outputs_size).to(device)

# Initialize weights for custom layers
nn.init.xavier_uniform_(input_embedding_layer.weight)
nn.init.xavier_uniform_(output_embedding_layer.weight)

# Define CTC loss function
loss_fn = CTCLoss(vocab_size)

# Project the softmaxed input to the embedding space
embedded_input = input_embedding_layer(softmaxed_input)

# Forward pass through the BERT model
outputs = model(inputs_embeds=embedded_input, attention_mask=attention_mask)
outputs = outputs.last_hidden_state

# Project the BERT output to the CTC vocab size
outputs = output_embedding_layer(outputs)
outputs = nn.functional.log_softmax(outputs, dim=-1)

# Calculate the CTC loss
loss = loss_fn(
    log_probs=outputs,
    targets=labels_batch,
    input_lengths=input_lengths,
    target_lengths=label_lengths
)

print(f'Softmaxed input shape: {softmaxed_input.shape}')
print(f'Output shape: {outputs.shape}')
print(f'Loss: {loss.item()}')
print('done!')
