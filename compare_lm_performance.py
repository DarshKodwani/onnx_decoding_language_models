import torch
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForCausalLM, AutoTokenizer

def pytorch_inference(torch_filepath, tokenizer, prompt, max_length=50, model_name="gpt2"):
    # Load pre-trained Hugging Face model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.load_state_dict(torch.load(torch_filepath))
    model.eval()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Perform inference
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length)
        end_time = time.time()

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return end_time - start_time, generated_text

def onnx_inference(onnx_filepath, tokenizer, prompt):
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_filepath)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_name = ort_session.get_inputs()[0].name
    input_ids = inputs["input_ids"].cpu().numpy()

    # Perform inference
    start_time = time.time()
    outputs = ort_session.run(None, {input_name: input_ids})

    # Decode the generated tokens
    logits = torch.tensor(outputs[0])
    predicted_ids = torch.argmax(logits, dim=-1)
    generated_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    end_time = time.time()
    return end_time - start_time, generated_text

def onnx_greedy_decode(onnx_filepath, tokenizer, prompt, max_length=10):
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_filepath)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy()

    # Perform greedy decoding
    generated_ids = input_ids
    start_time = time.time()
    for _ in range(max_length - input_ids.shape[1]):
        input_dict = {"input_ids": generated_ids}
        if "attention_mask" in {inp.name for inp in ort_session.get_inputs()}:
            input_dict["attention_mask"] = attention_mask

        # Get logits from ONNX
        logits = ort_session.run(None, input_dict)[0]

        # Take the token with the highest probability
        next_token = torch.tensor(logits[:, -1, :]).argmax(dim=-1, keepdim=True).numpy()

        # Append to generated sequence
        generated_ids = np.concatenate([generated_ids, next_token], axis=1)

        # Update attention mask
        attention_mask = np.concatenate(
            [attention_mask, np.ones((attention_mask.shape[0], 1))], axis=1
        )

    # Decode tokens into text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    end_time = time.time()
    return end_time - start_time, generated_text

if __name__ == "__main__":
    torch_filepath = "gpt2_model.pth"
    onnx_filepath = "gpt2_model.onnx"
    model_name = "gpt2"
    prompt = "The great scientist Isaac"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token to eos_token to handle padding
    tokenizer.pad_token = tokenizer.eos_token

    # PyTorch Inference
    pytorch_time, pytorch_generated = pytorch_inference(torch_filepath, tokenizer, prompt, max_length = 10, model_name='gpt2')
    print("------------------------------------------------------ ------------------------------------------------------")
    print("--------------------------------------------- PyTorch Inference ---------------------------------------------")
    print("------------------------------------------------------ ------------------------------------------------------")
    print(f"PyTorch Inference Time: {pytorch_time} seconds")
    print(f"\033[92mPyTorch Generated Text:\n{pytorch_generated}\n\033[0m")

    # ONNX Inference
    print("------------------------------------------------------ ------------------------------------------------------")
    print("---------------------------------------------  ONNX Inference   ---------------------------------------------")
    print("------------------------------------------------------ ------------------------------------------------------")
    onnx_inf_time, onnx_inf_generated = onnx_inference(onnx_filepath, tokenizer, prompt)
    print(f"ONNXI Inference Time: {onnx_inf_time} seconds")
    print(f"\033[92mONNX Generated Text:\n{onnx_inf_generated}\n \033[0m")
    
    # ONNX greedy Inference
    print("------------------------------------------------------ ------------------------------------------------------")
    print("---------------------------------------------  ONNX Greedy Inference   --------------------------------------")
    print("------------------------------------------------------ ------------------------------------------------------") 
    onnx_time, onnx_generated = onnx_greedy_decode(onnx_filepath, tokenizer, prompt)
    print(f"ONNX greedy Inference Time: {onnx_time} seconds")
    print(f"\033[92mONNX greedy Generated Text:\n{onnx_generated}\n \033[0m")
 
    # Compare Outputs
    print("Comparison of torch and greedy decoding outputs:")
    print(f"Are the outputs identical? {pytorch_generated == onnx_generated}")