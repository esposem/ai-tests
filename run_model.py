import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

#os.environ['TRANSFORMERS_CACHE'] = '/mnt/hugigingface_cache'

# Check if a GPU is available and select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("", flush=True)

# Load the FLAN-T5-Small model and tokenizer from Hugging Face
model_name = "google/flan-t5-small"
print(f"Downloading and loading model: {model_name}")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
model.to(device)

# Define the input text for the model
input_text = "Translate the following English text to French: 'Hello, how are you?'"
print("", flush=True)
print(f"Input text: {input_text}")

# Tokenize the input text
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate output from the model
print("Running inference...")
outputs = model.generate(input_ids, max_length=50)

# Decode the generated output and print
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated output: {decoded_output}")
