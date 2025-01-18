# ONNX Decoding Language Models

This repository provides tools and scripts for decoding language models using ONNX (Open Neural Network Exchange). It includes setup instructions, downloading GPT-2 model files, and running analysis scripts.

## Initial Setup

To set up the environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/onnx_decoding_language_models.git
    cd onnx_decoding_language_models
    ```

2. Run the first time setup file:
    ```bash
    ./firstTimeSetup.sh
    ```

3. Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```

## Downloading GPT-2 Files

To download the GPT-2 model files, run the following script:
```bash
python download_gpt2.py
```

This script will download the necessary GPT-2 model files and save them in the appropriate directory.

## Running the Analysis Script

To run the analysis script, use the following command:
```bash
python compare_lm_performance.py
```

This script will perform the analysis using the downloaded GPT-2 model files and output the results.
