# Igbo Basic Translator (Fine-Tune Project)

This repository contains the Google Colab notebook (`Fine_tune_LLM.ipynb`) used to fine-tune Microsoft's `Phi-3-mini-4k-instruct` on over 522,000 English-to-Igbo sentence pairs.

This project was my first dive into fine-tuning, resulting in a **specialist AI model**, not a general-purpose chatbot. The model is a **dedicated, one-way English-to-Igbo translation tool**. It excels at one task: responding to the prompt `Translate this English sentence to Igbo: '...'`.

**Final Model (GGUF):** [`nwokikeonyeka/igbo-phi3-translator`](https://huggingface.co/nwokikeonyeka/igbo-phi3-translator)

## 🚀 How to Test (Live Demo)

You can test the final, translated model directly in your browser using this simple, one-cell Google Colab notebook.

**Note:** This demo will install the model and run it on your Colab **CPU**. It may be slow (5 - 10 seconds per translation), but it is the simplest way to run it.

### One-Cell Colab Demo:

*Copy and paste this entire block into a single Google Colab cell.*

```python
# --- 1. Install the SIMPLE, CPU-only version ---
print("---  installing llama-cpp-python (CPU)... ---")
# This is the simple install. No build tools needed.
!pip install llama-cpp-python

print("\n--- ✅ All libraries installed! ---")

# --- 2. Import Libraries ---
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# --- 3. Define and Download Your Model ---
MODEL_NAME = "phi-3-mini-4k-instruct.Q4_K_M.gguf"
REPO_ID = "nwokikeonyeka/igbo-phi3-translator"

print(f"\n--- ⬇️ Downloading {MODEL_NAME} from {REPO_ID} ---")
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=MODEL_NAME
)
print(f"--- ✅ Model downloaded to: {model_path} ---")

# --- 4. Load the Model onto the CPU ---
print("--- 🧠 Loading model onto CPU... (This may take a moment) ---")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=0,  # 0 = Use CPU ONLY
    n_ctx=1024,      # Context size
    verbose=False    # Silence llama.cpp logs
)
print("--- ✅ Model loaded! Ready to test. ---")

# --- 5. Start the Interactive Test Loop ---
print("\n--- 🤖 Igbo Translator Test ---")
print("This model is a specialist. It only responds to the 'Translate' prompt.")
print("Type 'quit' or 'exit' to stop.")
print("-" * 40)

while True:
    try:
        user_prompt = str(input("English: "))
    except EOFError:
        break
        
    if user_prompt.lower() in ["quit", "exit"]:
        print("--- 👋 Test ended. ---")
        break
    if not user_prompt.strip():
        continue

    # 1. Format the prompt EXACTLY as it was trained
    full_prompt = f"Translate this English sentence to Igbo: '{user_prompt}'"
    formatted_input = f"<s>[INST] {full_prompt} [/INST]"

    print("Igbo AI: ...thinking...")

    # 2. Run Inference
    response = llm(
        formatted_input,
        max_tokens=100,          # Max length of the translation
        stop=["</s>", "[INST]"], # Stop when it finishes its response
        echo=False               # Don't repeat the prompt
    )
    
    # 3. Print the clean result
    try:
        ai_response = response["choices"][0]["text"].strip()
        print(f"Igbo AI: {ai_response}")
    except (IndexError, KeyError):
        print("Igbo AI: (No response generated. Make sure you typed in English.)")
```
-----

## Local Usage (Ollama / llama.cpp)

You can also run the final model locally.

**For Ollama:**

```bash
# Pull the model
ollama pull nwokikeonyeka/igbo-phi3-translator
  
# Run it
ollama run nwokikeonyeka/igbo-phi3-translator "Translate this English sentence to Igbo: 'Hello, how are you?'"
```

**For `llama.cpp`:**

```bash
./main -m ./phi-3-mini-4k-instruct.Q4_K_M.gguf -n 128 -p "<s>[INST] Translate this English sentence to Igbo: 'Hello, how are you?' [/INST]"
```

```
```

---

## Project Methodology: The "Colab Relay Race"

This model was trained for **32,646 steps**, a process that takes 24-48 hours of continuous GPU time. This is impossible to do in a single free Google Colab session, which typically times out after 3-6 hours.

The solution was the **"Colab Relay Race"**: a strategy for breaking one massive training job into many small, resumable parts.

### The Problem
Google Colab sessions are temporary. If your session times out at step 4,500, you lose all your progress.

### The Solution (The "Safety Net")
By using `push_to_hub=True` and `hub_strategy="checkpoint"` in the `TrainingArguments`, we use the **Hugging Face Hub as a permanent "safety net."**
1.  We set the script to save a checkpoint every `1000` steps.
2.  When a checkpoint is saved (e.g., at step 4,000), it is automatically uploaded to the `nwokikeonyeka/igbo-phi3-checkpoint` repo.
3.  When Colab inevitably times out, the last saved checkpoint (step 4,000) is safe.
4.  A new "worker" (or the same person after a cooldown) can start a new Colab session, run the "Resume" cell, and the trainer will securely download the step 4,000 checkpoint and **resume training from step 4,001**.
5.  This process is repeated until all 32,646 steps are complete.

---

## The Training Notebook (`Fine_tune_LLM.ipynb`)

The notebook in this repo is split into logical cells. The first cell sets up the environment, and the following cells are for executing the "Relay Race."

### Cell 1: Setup (Run by ALL Workers)
This cell installs libraries, logs into Hugging Face, loads the base `Phi-3-mini` model, loads the dataset, and initializes the `Trainer` with the correct 8*2 configuration. This must be run by every worker to ensure the settings match the checkpoint.

```python
# --- 1. Install Libraries ---
!pip install "unsloth[colab-new]" transformers peft bitsandbytes datasets trl

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import time
import os
from huggingface_hub import login, snapshot_download

# --- 2. HUGGING FACE LOGIN ---
login(token="YOUR_HF_TOKEN_GOES_HERE")
print("--- ✅ Hugging Face Login Successful ---")

# --- 3. Load the *BASE* Model ---
max_seq_length = 1024
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
print("--- ✅ Base model loaded ---")

# --- 4. Add PEFT/LoRA Adapters ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True, 
    random_state = 42,
    max_seq_length = max_seq_length,
)

# --- 5. Load and Format Your Dataset ---
full_dataset = load_dataset("ccibeekeoc42/english_to_igbo", split="train")

def format_example(example):
    if 'English' in example and 'Igbo' in example and \
       isinstance(example['English'], str) and isinstance(example['Igbo'], str):
        text = f"<s>[INST] Translate this English sentence to Igbo: '{example['English']}' [/INST] {example['Igbo']}</s>"
        return {"text": text}
    else:
        return {"text": None}

formatted_dataset = full_dataset.map(
    format_example,
    remove_columns=list(full_dataset.features),
    num_proc = 4
)

formatted_dataset = formatted_dataset.filter(lambda example: example.get("text") is not None)
print(f"--- Dataset loaded. Number of examples: {len(formatted_dataset)} ---")


# --- 6. The ORIGINAL, STABLE Training Arguments ---
training_args = TrainingArguments(
    max_steps = 32646,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,
    optim = "adamw_torch",
    learning_rate = 2e-5,
    lr_scheduler_type = "linear",
    save_strategy = "steps",
    save_steps = 1000,
    save_total_limit = 1,
    push_to_hub = True,
    hub_model_id = "nwokikeonyeka/igbo-phi3-checkpoint",
    hub_strategy = "checkpoint",
    logging_steps = 50,
    fp16 = True,
    group_by_length = True,
    report_to = "none",
)

# --- 7. Initialize the Trainer ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = training_args,
    packing = True,
)

print("--- ✅ All setup is complete! ---")
````

### Cell 2: Start Training (Worker 1 ONLY)

After running `Cell 1`, the *first worker* runs this cell to start the race from step 0.

```python
# Cell 2: Start the "Colab Relay Race" (Worker 1 ONLY)

print(f"--- 🚀 STARTING the 'Colab Relay Race' (Worker 1) ---")
print(f"--- This will run from step 0 and create the first checkpoint. ---")

start_time_train = time.time()

try:
    # Worker 1 just calls .train() to start from scratch
    trainer.train()
    print("\n--- 🎉 TRAINING COMPLETED NORMALLY! ---")

except Exception as e:
    print(f"\n--- 💥 Training interrupted by unexpected error: {e} ---")

finally:
    end_time_train = time.time()
    print(f"--- Training run duration: {(end_time_train - start_time_train) / 60:.2f} minutes ---")
    print("--- 🛑 Session ended. The first checkpoint is safe on Hugging Face Hub. ---")
```

### Cell 3: Resume Training (Worker 2, 3, 4...)

After running `Cell 1`, all subsequent workers run this cell to resume from the last saved checkpoint.

```python
# Cell 3: Resume the "Colab Relay Race" (Worker 2, 3...)

# --- Configuration ---
HUB_MODEL_ID = "nwokikeonyeka/igbo-phi3-checkpoint"
HUB_CHECKPOINT_SUBFOLDER = "last-checkpoint"
LOCAL_CHECKPOINT_PATH = os.path.join(os.path.expanduser("~"), "local_hub_resume")

print(f"--- 👟 RESUMING Training (Worker 2/3/...) ---")
print(f"--- ⬇️ Downloading checkpoint from Hub: {HUB_MODEL_ID}/{HUB_CHECKPOINT_SUBFOLDER} ---")

# --- 1. Download Checkpoint Files Locally ---
snapshot_download(
    repo_id=HUB_MODEL_ID,
    allow_patterns=[f"{HUB_CHECKPOINT_SUBFOLDER}/*"],
    local_dir=LOCAL_CHECKPOINT_PATH,
    local_dir_use_symlinks=False, # Use False for more stability in Colab
)

# --- 2. Define the Local Path to Resume From ---
RESUME_PATH = os.path.join(LOCAL_CHECKPOINT_PATH, HUB_CHECKPOINT_SUBFOLDER)
print(f"--- 🎯 Resuming from LOCAL PATH: {RESUME_PATH} ---")

# --- 3. Run the Training ---
start_time_train = time.time()
try:
    # This finds the last checkpoint and resumes training from that step
    trainer.train(resume_from_checkpoint = RESUME_PATH)
    print("\n--- 🎉 TRAINING COMPLETED NORMALLY! ---")

except Exception as e:
    print(f"\n--- 💥 Training interrupted by unexpected error: {e} ---")

finally:
    end_time_train = time.time()
    print(f"--- Training run duration: {(end_time_train - start_time_train) / 60:.2f} minutes ---")
    print("--- 🛑 Session ended. The latest checkpoint is safe on Hugging Face Hub. ---")
```
