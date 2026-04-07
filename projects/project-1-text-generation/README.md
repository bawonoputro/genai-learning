# Project 1: Text Generation with Pre-trained Models

## Overview
Experiments with 3 pre-trained language models from HuggingFace to understand how text generation works and how different parameters affect output.

### Models Tested
- **GPT-2** - Verbose, good quality
- **DistilGPT-2** - Fast but unstable with certain parameters
- **facebook/opt-350m** - Best balance of quality and stability

## Experiments

### 1. Multiple Models & Prompts Experimentation
3 models with 3 different prompts
- Findings: facebook/opt-350m produced most natural text

### 2. Temperature Experimentation
Temperature values: 0.5 and 1.5
- **Temperature 0.5**: Repetitive, safe, predictable
- **Temperature 1.5**: Creative but sometimes too random

### 3. top_k Experimentation
Tested top_k values: 1 (pick best word) and 5 (pick from 5 best words)
- **top_k=1**: Causes repetitive loops, avoid!
- **top_k=5**: Better diversity, more natural

## Key Findings

1. **Model Selection Matters**: Different models produce vastly different outputs
2. **Temperature Affects Randomness**: 
   - Low temp (0.5) = repetitive
   - High temp (1.5) = creative but sometimes it out of context
3. **top_k Prevents Loops**: Avoid top_k=1, causes model to get stuck
4. **Best Model**: `facebook/opt-350m`

## Sample Outputs
See results.txt for all experiment outputs.
