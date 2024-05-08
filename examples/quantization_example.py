import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setting seed for reproducibility
torch.manual_seed(42)

# Load DistilGPT2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Example text input
text = "My name is Ben,"
inputs = tokenizer(text, return_tensors='pt')

# Generate output before quantization with max length of 15 tokens
output = model.generate(**inputs, max_length=15)
print("Output before quantization:", tokenizer.decode(output[0]))

# Model size before quantization
original_model_size = sum(torch.numel(p) for p in model.parameters())
print("Model size before quantization:", original_model_size, "parameters")

# Applying quantization only to the linear layers
model.eval()  # Ensure the model is in evaluation mode
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Generate output after quantization with the same max length condition
quantized_output = quantized_model.generate(**inputs, max_length=15)
print("Output after quantization:", tokenizer.decode(quantized_output[0]))

# Model size after quantization (Note: This is just illustrative and might not show actual memory reduction)
quantized_model_size = sum(torch.numel(p) for p in quantized_model.parameters())
print("Model size after quantization:", quantized_model_size, "parameters")
