# Gemma-2-2B-IT-CyberAgent

🔗 **Model on Hugging Face**: [jprtr/gemma-2-2b-it-CyberAgent](https://huggingface.co/jprtr/gemma-2-2b-it-CyberAgent)

## Model Description

This is a fine-tuned version of google/gemma-2-2b-it, optimized for **on-device cybersecurity applications** for mobile devices. Unlike standard chatbots, this model is trained to output structured **JSON actions** (e.g., `scan_url`, `isolate_network`) that can be executed by an Android app or Edge AI Service.

The model has been adapted using **Supervised Fine-Tuning (SFT)** and **DPO (Direct Preference Optimization)** with **LoRA (Low-Rank Adaptation)** techniques to maintain high performance while remaining efficient for mobile and edge devices.

## Key Technologies

- **Unsloth**: Used for ultra-fast, memory-efficient fine-tuning (2x faster, 70% less memory)
- **LiteRT (formerly TFLite)**: Model format compatible with Google AI Edge Gallery for on-device inference
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning to keep the model lightweight

## Model Details

- **Base Model**: google/gemma-2-2b-it
- **Model Size**: 2 billion parameters (~2GB)
- **Model Type**: Causal Language Model (Gemma2ForCausalLM)
- **Fine-tuning Method**: LoRA + SFT + DPO
- **Optimization**: Mobile-first deployment
- **Precision**: bfloat16 / 4-bit quantization
- **Context Length**: 2048 tokens (training) / 8192 tokens (max)
- **Hardware Requirements**: GPU (L4/T4 recommended for training)

## Training

This model was fine-tuned with the following techniques:

### Supervised Fine-Tuning (SFT)

- **Training Steps**: 600 steps
- **Dataset**: Custom cybersecurity dataset with 2000+ threat examples
- **Focus**: Task-specific instruction tuning for security actions
- **Learning Rate**: 5e-5 (stable convergence)
- **Batch Size**: 2 with gradient accumulation (4 steps)

### DPO Training (Refining the Agent)

- **Training Steps**: 150 steps
- **Purpose**: Refine model responses for better alignment
- **Technique**: Direct Preference Optimization

### Data Preparation

- Clean synthetic dataset with EOS tokens
- Hard negatives for improved discrimination
- Structured JSON output format training

## Available Security Actions

The model can output these security actions:

- `scan_url(url)`: Check a link for phishing
- `kill_process(pid)`: Stop a suspicious app
- `isolate_network()`: Cut off internet access
- `ignore()`: No threat detected

## Input/Output Format

**Input**: Natural language threat description

**Output**: JSON action block

```json
{
  "thought": "Suspicious URL detected",
  "action": "scan_url",
  "params": {"url": "bit.ly/malware-site"}
}
```

## Implementation Workflow

This model outputs JSON action blocks that your application must parse and execute. Here's the complete workflow:

### 1. Model Generates JSON Instructions

When you send user input to the model (e.g., "Check this suspicious link: bit.ly/malware-site"), it analyzes the threat and outputs structured JSON:

```json
{
  "thought": "Suspicious URL detected",
  "action": "scan_url",
  "params": {"url": "bit.ly/malware-site"}
}
```

### 2. Application Parses JSON

Your Android app or Edge AI Service must:

- Parse the JSON response from the model
- Extract the `action` field to determine what security action to take
- Extract the `params` object to get necessary parameters (URL, process ID, etc.)
- Extract the `thought` field for logging/debugging

### 3. Execute Security Actions

Based on the action specified, your application implements the actual security function:

- **`scan_url(url)`**: Integrate with a URL scanning service (e.g., Google Safe Browsing API, VirusTotal) to check if the link is malicious
- **`kill_process(pid)`**: Use Android's `ActivityManager` or system APIs to terminate the suspicious application process
- **`isolate_network()`**: Disable network connectivity using `ConnectivityManager` or firewall APIs to prevent data exfiltration
- **`ignore()`**: No action needed - log the event and continue normal operation

**Important**: The model does NOT perform these actions itself. It only generates the instructions. Your application must implement the actual security mechanisms.

## Usage

### Python

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "jprtr/gemma-2-2b-it-CyberAgent"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Security agent prompt
agent_prompt = """You are an autonomous security agent on a Pixel device.
Analyze the user's input. If a threat is detected, output a JSON action block.
Available Actions:
- scan_url(url): Check a link for phishing.
- kill_process(pid): Stop a suspicious app.
- isolate_network(): Cut off internet access.
- ignore(): No threat found.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

input_text = "Check this suspicious link: bit.ly/malware-site"
prompt = agent_prompt.format(input_text, "", "")
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
response = tokenizer.batch_decode(outputs)[0].split("### Response:")[1].strip()
print(response)
```

## Training Notebook

The complete training pipeline is available in this repository:

- **Notebook**: Production-ready Google Colab notebook with full training workflow
- **File**: `Gemma_2_2B_Cybersecurity_Agent_Mobile.ipynb`
- See the notebook for complete step-by-step training instructions

## Intended Use

- Mobile and edge device cybersecurity
- On-device AI security applications
- Autonomous threat detection and response
- Resource-constrained environments
- Android security agents
- Privacy-focused local inference

## Performance

- **Training Time**: ~1-2 hours on L4 GPU
- **Model Size**: ~2GB (suitable for modern Android devices with 6GB+ RAM)
- **Inference Speed**: Optimized for on-device execution
- **Memory Efficiency**: 70% less memory usage with Unsloth optimization

## Limitations

- This model inherits the limitations of the base Gemma 2-2B model
- Optimized for mobile deployment, performance may vary on different hardware
- As with all language models, outputs should be verified for accuracy
- AI Edge Torch conversion had compatibility issues - use PyTorch Mobile or ONNX Runtime instead
- Trained specifically for cybersecurity actions - not a general-purpose chatbot

## Deployment Options

1. **PyTorch Mobile** (recommended for Android)
2. **ONNX Runtime Mobile**
3. **TensorFlow Lite** (via ONNX conversion)

## Citation

If you use this model, please cite both the original Gemma model and this fine-tuned version:

```bibtex
@misc{gemma-2-2b-it-cyberagent,
  author = {CyberAgent},
  title = {Gemma-2-2B-IT-CyberAgent: Mobile Cybersecurity Agent},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/jprtr/gemma-2-2b-it-CyberAgent}
}
```

## License

This model is released under the Gemma license. See the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) for more details.
