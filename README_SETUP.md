# LLM Few-shot Multivariate Time Series Classification - Modified for MacBook Air M4

This project has been modified to work with Groq API instead of local LLMs, making it compatible with MacBook Air M4.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure your API key:
```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

To get a Groq API key:
1. Visit https://console.groq.com/
2. Sign up for an account
3. Generate an API key
4. Add it to your `.env` file

### 3. Data Preparation

The project expects data in `./npydata/` directory. For the Handwriting dataset:
- `Handwriting_train_x.npy`
- `Handwriting_train_y.npy`
- `Handwriting_test_x.npy`
- `Handwriting_test_y.npy`

### 4. Running the Model

#### Option 1: Local LLM Wrapper (No API Required)
For quick testing without API costs:
```bash
./run_local.sh
```

#### Option 2: Groq API-based LLM
For using actual Groq models (requires API key):
```bash
./run_api.sh
```

#### Option 3: Manual Execution
```bash
python main.py --llm_type groq-llama3-8b --dataset Handwriting
```

## Available LLM Types

### Groq API Models (require GROQ_API_KEY)
- `groq-llama3-8b`: Llama 3.1 8B Instant model
- `groq-llama3-70b`: Llama 3.1 70B Versatile model  
- `groq-mixtral`: Mixtral 8x7B model
- `groq-gemma`: Gemma2 9B model

### Local Models (no API required)
- `local-small`: Small local transformer (512 dim)
- `local-medium`: Medium local transformer (768 dim)
- `local-large`: Large local transformer (1024 dim)

## Configuration

Edit `llm_config.json` to modify model parameters like dimensions and number of layers.

## Notes

- The Groq API wrapper converts time series data to text representations for the language model
- Local wrappers use transformer layers for efficient computation without API calls
- Batch size may need to be reduced when using API models due to rate limits
- The project now works on Mac Silicon (M1/M2/M4) without CUDA dependencies

## Troubleshooting

1. **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
2. **API errors**: Check that your GROQ_API_KEY is correctly set in `.env`
3. **Memory issues**: Reduce batch size in the run scripts
4. **CUDA errors**: The project now uses CPU/MPS automatically on Mac

## Original Features Maintained

- Few-shot learning capabilities
- Multiple datasets support
- Patch embedding for time series
- Causal CNN encoder
- LoRA fine-tuning (for local models)
