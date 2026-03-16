import json
import torch
from chatbot import load_model, get_prompt, MAX_LENGTH, MAX_NEW_TOKENS
from evaluation_framework import EvaluationRunner


def run_evaluation():
    """Generate responses from queries and evaluate them"""
    model = load_model()
    with open("queries.json", "r") as f:
        queries = json.load(f)

    history = []
    results = []

    print("Generating responses...\n")
    for item in queries:
        user_input = item.get("user_input", "")
        if not user_input:
            continue

        full_prompt = get_prompt(history, user_input)
        input_tokens = model.tokenizer(
            full_prompt,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        input_ids = input_tokens['input_ids'].cuda()

        generation_output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True,
        )

        output_sequence = generation_output.sequences[0]
        full_response = model.tokenizer.decode(output_sequence, skip_special_tokens=True)
        if "Assistant:" in full_response:
            response_text = full_response.split("Assistant:")[-1].strip()
        else:
            response_text = full_response.replace(full_prompt, "").strip()

        results.append({"user_input": user_input, "response": response_text})

        history.append({"user": user_input, "assistant": response_text})
        if len(history) > 10:
            history.pop(0)

        torch.cuda.empty_cache()

    with open("responses.json", "w") as out:
        json.dump(results, out, indent=2)
    
    print("Responses saved to responses.json")
    print("\nRunning evaluation framework...\n")
    
    # Run evaluation framework
    EvaluationRunner.run_evaluation()


if __name__ == "__main__":
    run_evaluation()
