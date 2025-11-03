# Basic Text Tokenization Implementation
# This module provides two tokenizer implementations:
# 1. SimpleTokenizerV1: Basic tokenizer that splits text into words and punctuation
# 2. SimpleTokenizerV2: Enhanced tokenizer that handles unknown words with special tokens
#
# The tokenizers support:
# - Splitting text into individual tokens (words and punctuation)
# - Converting tokens to and from integer IDs
# - Handling basic punctuation and whitespace
# - Special token support (in V2)
#
# Implementation follows a process of:
# 1. Splitting text using regular expressions
# 2. Mapping tokens to integer IDs using a vocabulary
# 3. Providing methods to encode (text to IDs) and decode (IDs to text)

# Import the regular expression module for text splitting operations
import re

class SimpleTokenizerV1:
    """
    A basic tokenizer that converts text to token IDs and back.
    
    This tokenizer:
    1. Splits text on punctuation and whitespace
    2. Converts tokens to integer IDs using a provided vocabulary
    3. Reconstructs text from token IDs with proper spacing
    """
    
    def __init__(self, vocab):
        """
        Initialize the tokenizer with a vocabulary.
        
        Args:
            vocab (dict): A dictionary mapping token strings to integer IDs
                          Example: {"hello": 0, "world": 1, ".": 2}
        """
        # Store the string-to-integer mapping (vocabulary)
        self.str_to_int = vocab
        
        # Create the reverse mapping (integer-to-string) by swapping keys and values
        # This is used during decoding to convert IDs back to tokens
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        """
        Convert text into a sequence of token IDs.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            list: A list of integer token IDs
            
        Process:
        1. Split text using regex pattern that captures punctuation and whitespace
        2. Remove empty strings and strip whitespace from tokens
        3. Convert each token to its corresponding integer ID
        """
        # Split the text using a regex pattern that matches:
        # - Punctuation marks: , . : ; ? _ ! " ( ) '
        # - Double dashes: --
        # - Whitespace: \s
        # The parentheses in the regex pattern capture these separators as tokens
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        # Filter out empty strings and strip whitespace from each token
        # This creates a clean list of tokens without empty elements
        preprocessed = [ item.strip() for item in preprocessed if item.strip() ]

        # Convert each token string to its corresponding integer ID using the vocabulary
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        """
        Convert a sequence of token IDs back into text.
        
        Args:
            ids (list): A list of integer token IDs
            
        Returns:
            str: The reconstructed text
            
        Process:
        1. Convert each ID to its corresponding token string
        2. Join tokens with spaces
        3. Fix spacing around punctuation for natural text appearance
        """
        # Convert each integer ID back to its string token and join with spaces
        text = " ".join([self.int_to_str[i] for i in ids])
        
        # Remove spaces before punctuation marks for natural text appearance
        # This regex finds spaces followed by punctuation and replaces with just the punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# Step 2: Creating Token IDs with special tokens
# Enhanced tokenizer that handles unknown words and special tokens

# Enhanced Tokenizer with Special Token Support
# This version adds support for:
# - Unknown word handling using <|unk|> token
# - Support for special tokens like <|endoftext|>
# - Improved text reconstruction with proper spacing
#
# Special tokens supported:
# - <|unk|>: For unknown words not in vocabulary
# - <|endoftext|>: For separating different texts
# - [BOS]: Beginning of sequence (optional)
# - [EOS]: End of sequence (optional)
# - [PAD]: For padding sequences to same length (optional)

class SimpleTokenizerV2:
    """
    An enhanced tokenizer that handles unknown words and special tokens.
    
    Improvements over SimpleTokenizerV1:
    1. Handles out-of-vocabulary (OOV) words by replacing them with <|unk|> token
    2. Supports special tokens for sequence management
    3. Maintains the same interface for encoding and decoding
    """
    
    def __init__(self, vocab):
        """
        Initialize the enhanced tokenizer with a vocabulary.
        
        Args:
            vocab (dict): A dictionary mapping token strings to integer IDs
                          Should include special tokens like <|unk|>
        
        Note:
            The vocabulary should include the <|unk|> token and any other
            special tokens that will be used (e.g., <|endoftext|>, [BOS], [EOS], [PAD])
        """
        # Store the string-to-integer mapping (vocabulary)
        self.str_to_int = vocab
        
        # Create the reverse mapping (integer-to-string) for decoding
        # This swaps keys and values from the original vocabulary
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        """
        Convert text into a sequence of token IDs with unknown word handling.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            list: A list of integer token IDs
            
        Process:
        1. Split text using regex pattern for punctuation and whitespace
        2. Remove empty strings and strip whitespace
        3. Replace unknown tokens (not in vocabulary) with <|unk|> token
        4. Convert tokens to integer IDs
        """
        # Split the text using the same regex pattern as SimpleTokenizerV1
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        
        # Filter out empty strings and strip whitespace
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # Handle unknown words by replacing them with <|unk|> token
        # This is a key improvement over SimpleTokenizerV1
        # The list comprehension checks if each token exists in the vocabulary:
        # - If it does, keep the original token
        # - If not, replace it with the <|unk|> token
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        # Convert each token to its integer ID
        # Since we've replaced unknown tokens with <|unk|>, all tokens are now in the vocabulary
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        """
        Convert a sequence of token IDs back into text.
        
        Args:
            ids (list): A list of integer token IDs
            
        Returns:
            str: The reconstructed text
            
        Process:
        1. Convert each ID to its corresponding token string
        2. Join tokens with spaces
        3. Fix spacing around punctuation for natural text appearance
        
        Note:
            Special tokens like <|unk|> will appear in the output text
            as they are part of the vocabulary
        """
        # Convert each integer ID back to its string token and join with spaces
        text = " ".join([self.int_to_str[i] for i in ids])
        
        # Remove spaces before punctuation marks for natural text appearance
        # Same as in SimpleTokenizerV1
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# Note on special tokens:
# [BOS] - Beginning of sequence token: Marks the start of a sequence, useful for models to recognize the beginning
# [EOS] - End of sequence token: Marks the end of a sequence, helps models know when to stop generating
# [PAD] - Padding token: Used to make sequences the same length in a batch, typically ignored in loss calculation
# <|endoftext|> - Used by GPT models: Indicates the end of a document or separate sections of text
# <|unk|> - Used for unknown words: Represents words not found in the vocabulary
