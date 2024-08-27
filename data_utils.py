import torch
import random

def generate_batch(batch_size, max_input_length, vocab_size):
    """
    Generates a batch of inputs with variable lengths and corresponding labels.
    Adjusts the input batch size to the maximum real length encountered.
    
    Args:
    - batch_size (int): The number of samples in the batch.
    - max_input_length (int): The maximum length of the input sequences.
    - vocab_size (int): The size of the vocabulary.
    
    Returns:
    - input_batch (torch.Tensor): A batch of input sequences of shape (batch_size, max_real_length, vocab_size + 1).
    - labels_batch (torch.Tensor): A batch of label sequences of shape (batch_size, max_real_length).
    - input_lengths (torch.Tensor): The lengths of the input sequences.
    - label_lengths (torch.Tensor): The lengths of the label sequences.
    """
    ctc_outputs_size = vocab_size + 1  # Including CTC blank token

    input_batch = []
    labels_batch = []
    input_lengths = []
    label_lengths = []

    for _ in range(batch_size):
        input_length = random.randint(1, max_input_length)
        label_length = random.randint(1, input_length)

        # Generate random input
        input_sequence = torch.rand(input_length, ctc_outputs_size) * 2 - 1
        # Softmaxed input
        input_sequence = torch.nn.functional.log_softmax(input_sequence, dim=-1)
        
        # Generate random labels
        label_sequence = torch.randint(0, vocab_size, (label_length,))

        input_batch.append(input_sequence)
        labels_batch.append(label_sequence)
        input_lengths.append(input_length)
        label_lengths.append(label_length)

    # Convert to tensors
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=0)
    labels_batch = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-1)
    
    input_lengths = torch.tensor(input_lengths)
    label_lengths = torch.tensor(label_lengths)

    # Sort by input lengths in descending order
    sorted_lengths, sorted_indices = torch.sort(input_lengths, descending=True)
    input_batch = input_batch[sorted_indices]
    labels_batch = labels_batch[sorted_indices]
    label_lengths = label_lengths[sorted_indices]

    # Trim input and labels to the maximum real length
    max_real_length = sorted_lengths[0].item()
    input_batch = input_batch[:, :max_real_length, :]
    labels_batch = labels_batch[:, :max_real_length]

    return input_batch, labels_batch, sorted_lengths, label_lengths

if __name__ == '__main__':
    # Example usage
    batch_size = 32
    max_input_length = 512
    vocab_size = 128

    input_batch, labels_batch, input_lengths, label_lengths = generate_batch(batch_size, max_input_length, vocab_size)

    print(f'Input batch shape: {input_batch.shape}')
    print(f'Labels batch shape: {labels_batch.shape}')
    print(f'Input lengths: {input_lengths}')
    print(f'Label lengths: {label_lengths}')
    print('done!')
