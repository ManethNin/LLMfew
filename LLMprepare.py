import json
import torch
import os
from dotenv import load_dotenv
from APILLMWrapper import APILLMWrapper, SimpleLLMWrapper

# Load environment variables
load_dotenv()

def LLMprepare(configs):
    # Load model configurations from JSON file
    with open('llm_config.json', 'r') as f:
        model_configurations = json.load(f)

    config = model_configurations.get(configs.llm_type)
    if not config:
        raise ValueError(f"Unsupported LLM type: {configs.llm_type}")

    d_model = config["dim"]
    use_api = config.get("use_api", False)
    
    if use_api and os.getenv('GROQ_API_KEY'):
        # Use API-based LLM wrapper
        llm_model = APILLMWrapper(
            model_name=config.get("model_name", "llama-3.1-8b-instant"),
            embedding_dim=d_model
        )
        print(f"Using Groq API-based LLM: {config.get('model_name', 'llama-3.1-8b-instant')}")
    else:
        # Use simple local LLM wrapper (more efficient for training)
        llm_model = SimpleLLMWrapper(
            input_dim=d_model,
            hidden_dim=d_model,
            num_layers=config.get("num_layers", 6)
        )
        print(f"Using local LLM wrapper with {d_model} dimensions")
    
    # Handle LoRA configuration for local models only
    if configs.lora and not use_api:
        # For SimpleLLMWrapper, we can make specific parameters trainable
        for name, param in llm_model.named_parameters():
            if 'transformer' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        # For API models or non-LoRA, make all parameters trainable
        for param in llm_model.parameters():
            param.requires_grad = True

    return llm_model, d_model
