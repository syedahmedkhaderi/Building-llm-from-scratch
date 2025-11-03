# BYTE PAIR ENCODING
# This module implements the BPE (Byte Pair Encoding) tokenization approach,
# which is used by models like GPT to handle unknown words by breaking them into subword units.
# It also includes data loading and embedding functionality for machine learning models.

# The implementation includes:
# 1. A Dataset class for handling tokenized text with sliding windows
# 2. A DataLoader creation utility
# 3. Token and positional embedding utilities

# Import required libraries:
# - torch: Main PyTorch library for tensor operations and neural networks
# - Dataset, DataLoader: PyTorch classes for data handling
# - tiktoken: OpenAI's BPE tokenizer implementation
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    """
    PyTorch Dataset for handling text data for language model training.
    
    This dataset:
    1. Tokenizes text using BPE tokenization (via tiktoken)
    2. Creates overlapping chunks of tokens using a sliding window approach
    3. Prepares input-target pairs for language modeling (predicting next token)
    
    The sliding window approach allows efficient processing of long texts
    by breaking them into manageable chunks while maintaining context.
    """
    
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Initialize the dataset with text and tokenization parameters.
        
        Args:
            txt (str): The input text to tokenize and process
            tokenizer: The tokenizer object (tiktoken encoder)
            max_length (int): Maximum sequence length for each chunk
            stride (int): Number of tokens to slide the window by
                          Smaller stride = more overlap between chunks
        
        Process:
        1. Tokenize the entire text at once
        2. Create overlapping chunks using sliding window
        3. For each chunk, create input and target sequences
           (target is input shifted by one position)
        """
        # Initialize empty lists to store input and target token sequences
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text at once using the provided tokenizer
        # allowed_special parameter tells the tokenizer to keep the <|endoftext|> token
        # This returns a list of integer token IDs
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window approach to create overlapping chunks
        # The window size is max_length and moves by stride tokens each step
        # This creates multiple training examples from a single text
        for i in range(0, len(token_ids) - max_length, stride):
            # Extract the current chunk of tokens (input sequence)
            input_chunk = token_ids[i:i + max_length]
            
            # Create the target sequence by shifting input by one position
            # This is for next-token prediction: given tokens [0...n-1], predict token n
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            # Convert token ID lists to PyTorch tensors and store them
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        Return the number of chunks (samples) in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Get a single input-target pair by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (input_tensor, target_tensor) pair for training
                   Both are 1D tensors of token IDs
        """
        return self.input_ids[idx], self.target_ids[idx]

# Data Loader Creation
# This function creates a DataLoader that:
# - Uses sliding windows to chunk text into sequences
# - Supports batch processing
# - Allows for stride configuration for overlapping sequences
# - Uses the GPT-2 tokenizer from tiktoken

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                        stride=128, shuffle=True, drop_last=True,
                        num_workers=0):
    """
    Create a PyTorch DataLoader for language model training.
    
    This function:
    1. Initializes a GPT-2 tokenizer from tiktoken
    2. Creates a dataset with the provided text
    3. Wraps the dataset in a DataLoader for batch processing
    
    Args:
        txt (str): The input text to tokenize and process
        batch_size (int): Number of samples per batch (default: 4)
        max_length (int): Maximum sequence length for each chunk (default: 256)
        stride (int): Number of tokens to slide the window by (default: 128)
                      Controls overlap between chunks
        shuffle (bool): Whether to shuffle the data (default: True)
        drop_last (bool): Whether to drop the last incomplete batch (default: True)
        num_workers (int): Number of subprocesses for data loading (default: 0)
                           0 means data loading happens in the main process
    
    Returns:
        DataLoader: PyTorch DataLoader object for training
        
    Note:
        - Smaller stride values create more training examples with more overlap
        - Setting shuffle=True is important for training to prevent the model
          from learning the order of chunks
        - drop_last=True prevents issues with incomplete batches
    """
    # Initialize the GPT-2 tokenizer from tiktoken
    # This tokenizer uses Byte Pair Encoding (BPE) to split text into subword units
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create a dataset instance using our custom GPTDatasetV1 class
    # This handles tokenization and sliding window chunking
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create a PyTorch DataLoader to handle batching and other data loading features
    # - dataset: The dataset to load data from
    # - batch_size: Number of samples per batch
    # - shuffle: Whether to reshuffle data at every epoch
    # - drop_last: Whether to drop the last incomplete batch
    # - num_workers: How many subprocesses to use for data loading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# Token and Positional Embeddings
# This section implements the embedding layers needed for transformer models:
# 1. Token embeddings: Convert token IDs into dense vectors
# 2. Positional embeddings: Encode position information
# 3. Combined embeddings: Add token and positional embeddings together
#
# These embeddings are crucial for:
# - Capturing semantic meaning of tokens
# - Preserving sequence order information
# - Providing input representations for the model

def create_embeddings(vocab_size, output_dim, max_length):
    """
    Create token and positional embedding layers for a transformer model.
    
    This function creates two embedding layers:
    1. Token embedding: Maps token IDs to dense vectors
    2. Positional embedding: Maps position indices to dense vectors
    
    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens)
        output_dim (int): Dimension of the embedding vectors
        max_length (int): Maximum sequence length to support
    
    Returns:
        tuple: (token_embedding_layer, pos_embedding_layer)
               Both are torch.nn.Embedding instances
    
    Note:
        - Token embeddings capture semantic meaning of words
        - Positional embeddings capture sequence order information
        - Both embeddings have the same dimension (output_dim) to allow addition
    """
    # Create token embedding layer
    # This layer maps each token ID (integer) to a dense vector of size output_dim
    # The embedding table has vocab_size rows (one for each possible token)
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    
    # Create positional embedding layer
    # This layer maps each position index (0, 1, 2, ...) to a dense vector
    # The embedding table has max_length rows (one for each possible position)
    pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
    
    return token_embedding_layer, pos_embedding_layer

def get_embeddings(token_ids, token_embedding_layer, pos_embedding_layer):
    """
    Get combined token and positional embeddings for input tokens.
    
    This function:
    1. Converts token IDs to token embeddings
    2. Creates position indices (0, 1, 2, ...) for the sequence
    3. Converts position indices to positional embeddings
    4. Adds token and positional embeddings together
    
    Args:
        token_ids (torch.Tensor): Batch of token ID sequences [batch_size, seq_length]
        token_embedding_layer (torch.nn.Embedding): Token embedding layer
        pos_embedding_layer (torch.nn.Embedding): Positional embedding layer
    
    Returns:
        torch.Tensor: Combined embeddings [batch_size, seq_length, embedding_dim]
    
    Note:
        - The addition of token and positional embeddings is a key feature of transformer models
        - This allows the model to understand both the meaning of tokens and their positions
        - The result is used as input to the transformer encoder/decoder layers
    """
    # Convert token IDs to token embeddings
    # Input: [batch_size, seq_length]
    # Output: [batch_size, seq_length, embedding_dim]
    token_embeddings = token_embedding_layer(token_ids)
    
    # Create position indices (0, 1, 2, ...) for the sequence length
    # torch.arange creates a 1D tensor with values [0, 1, 2, ..., seq_length-1]
    # This will be used to get embeddings for each position in the sequence
    positions = torch.arange(token_ids.shape[1])
    
    # Convert position indices to positional embeddings
    # Input: [seq_length]
    # Output: [seq_length, embedding_dim]
    pos_embeddings = pos_embedding_layer(positions)
    
    # Add token and positional embeddings
    # The positional embeddings are automatically broadcast across the batch dimension
    # Output: [batch_size, seq_length, embedding_dim]
    return token_embeddings + pos_embeddings
