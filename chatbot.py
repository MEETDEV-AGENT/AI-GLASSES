import sys
import torch
from airllm import AutoModel

# ================= CONFIGURATION =================
# WARNING: 105B is extremely large. 
# Ensure you have enough Disk Space (~200GB+) and System RAM (64GB+).
MODEL_ID = "sarvamai/sarvam-105b" 
MAX_LENGTH = 2048  # Context window limit
MAX_NEW_TOKENS = 150 # Response length
# =================================================

def load_model():
    print(f"Loading {MODEL_ID} with AirLLM...")
    print("This may take several minutes depending on your internet and disk speed.")
    try:
        # AirLLM automatically handles layer offloading to fit VRAM
        model = AutoModel.from_pretrained(MODEL_ID)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Tip: Ensure you are logged in to Hugging Face if the model is gated.")
        print("Tip: Check your disk space.")
        sys.exit(1)

def get_prompt(history, user_input):
    """
    Formats the conversation history into a prompt string.
    Note: You may need to adjust this template based on Sarvam's specific training format.
    """
    # Generic Instruction Format
    prompt = ""
    for entry in history:
        prompt += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n"
    
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

def chat_loop(model):
    history = []
    print("\n--- Sarvam 105B Chatbot Started ---")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue

            # 1. Format Prompt with History
            full_prompt = get_prompt(history, user_input)

            # 2. Tokenize
            # We tokenize the entire history + new input to maintain context
            input_tokens = model.tokenizer(
                full_prompt,
                return_tensors="pt",
                return_attention_mask=False,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False
            )

            # 3. Move to Device
            input_ids = input_tokens['input_ids'].cuda()

            # 4. Generate
            print("Sarvam: ", end="", flush=True)
            
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=model.tokenizer.eos_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )

            # 5. Decode Output
            output_sequence = generation_output.sequences[0]
            full_response = model.tokenizer.decode(output_sequence, skip_special_tokens=True)

            # 6. Extract only the new response (remove prompt history)
            # We split by the last "Assistant:" tag to get just the generated text
            if "Assistant:" in full_response:
                response_text = full_response.split("Assistant:")[-1].strip()
            else:
                response_text = full_response.replace(full_prompt, "").strip()

            print(response_text)
            print("\n") # Newline for next input

            # 7. Update History
            history.append({
                "user": user_input,
                "assistant": response_text
            })

            # Optional: Limit history to prevent context overflow
            if len(history) > 10:
                history.pop(0)

            # Clear CUDA cache occasionally to prevent fragmentation
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Try reducing MAX_LENGTH or MAX_NEW_TOKENS.")
            break

if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not detected. AirLLM works best with an NVIDIA GPU.")
    
    model = load_model()
    chat_loop(model)
