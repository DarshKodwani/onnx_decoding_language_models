from transformers import GPT2Model, GPT2Tokenizer
import torch
from pathlib import Path
from transformers.convert_graph_to_onnx import convert

def download_and_save_gpt2(pytorch_save_path, onnx_save_path):
    # Create directories if they don't exist
    Path(pytorch_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(onnx_save_path).parent.mkdir(parents=True, exist_ok=True)

    # Load GPT-2 model and tokenizer
    model = GPT2Model.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Save PyTorch model
    torch.save(model.state_dict(), pytorch_save_path)
    print(f"Saved PyTorch model to {pytorch_save_path}")

    # Export model to ONNX
    print(f"Converting to ONNX format...")
    convert(
        framework="pt",
        model=model,
        tokenizer=tokenizer,
        output=onnx_save_path,
        opset=11,  # ONNX opset version
        pipeline_name="text-generation",
    )
    print(f"Saved ONNX model to {onnx_save_path}")

if __name__ == "__main__":
    # File paths for the model
    pytorch_model_path = "gpt2_model.pth"
    onnx_model_path = "gpt2_model.onnx"

    download_and_save_gpt2(pytorch_model_path, onnx_model_path)