import os
import torch
import torch.nn as nn
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class APILLMWrapper(nn.Module):
    """
    A wrapper that simulates local LLM behavior using Groq API-based models.
    Since Groq doesn't have dedicated embedding models, we use chat completions
    to generate fixed-size representations for time series data.
    """
    
    def __init__(self, model_name="llama-3.1-8b-instant", embedding_dim=1536):
        super(APILLMWrapper, self).__init__()
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Create a learnable projection layer to map embeddings to desired dimension
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Cache for embeddings to avoid repeated API calls for same data
        self.embedding_cache = {}
        
    def get_text_embedding(self, text):
        """
        Get embedding-like representation for text using Groq API.
        Since Groq doesn't have embedding endpoints, we use a chat completion
        to generate a consistent numerical representation.
        """
        # Check cache first
        text_hash = hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
            
        try:
            # Use chat completion to generate a consistent representation
            prompt = f"""
            Convert this time series data into a fixed-length numerical vector representation.
            Return exactly {self.embedding_dim} floating point numbers separated by commas.
            Time series data: {text[:500]}  # Limit text length for API
            
            Output format: number1,number2,number3,...
            """
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=self.embedding_dim * 3  # Enough tokens for numbers
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Parse the response to extract numbers
            try:
                # Extract numbers from response
                numbers = []
                for part in response_text.split(','):
                    try:
                        num = float(part.strip())
                        numbers.append(num)
                    except ValueError:
                        continue
                
                # If we don't have enough numbers, pad with random values
                while len(numbers) < self.embedding_dim:
                    numbers.append(np.random.normal(0, 0.1))
                
                # If we have too many, truncate
                numbers = numbers[:self.embedding_dim]
                
                embedding = np.array(numbers, dtype=np.float32)
                
            except Exception as e:
                print(f"Error parsing Groq response, using random embedding: {e}")
                embedding = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
            
            # Cache the result
            self.embedding_cache[text_hash] = embedding
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding from Groq: {e}")
            # Return random embedding as fallback
            embedding = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
            self.embedding_cache[text_hash] = embedding
            return embedding
    
    def forward(self, inputs_embeds):
        """
        Forward pass that simulates LLM behavior.
        inputs_embeds: tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        
        # For each sequence in the batch, convert to text representation and get embedding
        embeddings = []
        
        for i in range(batch_size):
            # Convert the numerical sequence to a text representation
            sequence = inputs_embeds[i].detach().cpu().numpy()
            
            # Create a simple text representation of the sequence
            # Use statistics to create a more compact representation
            mean_vals = np.mean(sequence, axis=0)
            std_vals = np.std(sequence, axis=0)
            max_vals = np.max(sequence, axis=0)
            min_vals = np.min(sequence, axis=0)
            
            text_repr = f"Time series statistics - Mean: {mean_vals[:10].tolist()}, Std: {std_vals[:10].tolist()}, Max: {max_vals[:10].tolist()}, Min: {min_vals[:10].tolist()}"
            
            # Get embedding from API
            embedding = self.get_text_embedding(text_repr)
            
            # Repeat embedding for each position in sequence
            seq_embeddings = np.tile(embedding, (seq_len, 1))
            embeddings.append(seq_embeddings)
        
        # Convert to tensor
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=inputs_embeds.device)
        
        # Apply learned projection
        embeddings_tensor = self.projection(embeddings_tensor)
        
        # Return in the format expected by the model (similar to transformers output)
        return type('obj', (object,), {'last_hidden_state': embeddings_tensor})


class SimpleLLMWrapper(nn.Module):
    """
    A simpler wrapper that uses local neural networks to simulate LLM behavior
    without requiring API calls. This is more efficient for training.
    """
    
    def __init__(self, input_dim, hidden_dim=1536, num_layers=6):
        super(SimpleLLMWrapper, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Create a simple transformer-like architecture
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, inputs_embeds):
        """
        Forward pass that simulates LLM behavior using local computation.
        inputs_embeds: tensor of shape (batch_size, seq_len, input_dim)
        """
        # Project to hidden dimension
        x = self.input_projection(inputs_embeds)
        
        # Apply transformer layers
        x = self.transformer(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Return in the format expected by the model
        return type('obj', (object,), {'last_hidden_state': x})
