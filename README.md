# Igbo Basic Translator (Fine-Tune Project)

This repository contains the Google Colab notebook (`.ipynb`) used to fine-tune Microsoft's `Phi-3-mini-4k-instruct` on over 522,000 English-to-Igbo sentence pairs.

This project was a deep dive into fine-tuning, resulting in a **specialist AI model**, not a general-purpose chatbot. The model is a **dedicated, one-way English-to-Igbo translation tool**. It excels at one task: responding to the prompt `Translate this English sentence to Igbo: '...'`.

The notebook `Phase_1_2_Train_LLM.ipynb` contains the full "Colab Relay Race" script that was used to train the model over 32,000 steps.

**Final Model (GGUF):** [`nwokikeonyeka/igbo-phi3-translator`](https://huggingface.co/nwokikeonyeka/igbo-phi3-translator)

---

## 🚀 How to Test (Live Demo)

You can test the final, translated model directly in your browser using this simple, one-cell Google Colab notebook.

**Note:** This demo will install the model and run it on your Colab **CPU**. It may be slow (10 seconds per translation), but it is the simplest way to run it.

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
