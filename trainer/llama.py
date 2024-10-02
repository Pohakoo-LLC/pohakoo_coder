import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from validator import is_correct
from accelerate import Accelerator  # Import Accelerator for distributed training
import json

# Initialize the Accelerator
accelerator = Accelerator(mixed_precision="bf16")  # You can also use 'fp16' depending on hardware support

# Load pre-trained CodeLlama model and tokenizer
model_name = 'meta-llama/CodeLlama-70b-Python-hf'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Set padding token to be the same as EOS token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Prepare the model and optimizer using Accelerator
optimizer = AdamW(model.parameters(), lr=5e-5)
model, optimizer = accelerator.prepare(model, optimizer)  # Prepare model and optimizer for accelerator

# Function to generate responses and calculate rewards
def generate_and_evaluate(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    inputs = accelerator.prepare(inputs)  # Ensure the inputs are prepared for Accelerator
    
    # Generate text using the model
    outputs = model.generate(inputs.input_ids, max_length=250, attention_mask=inputs.attention_mask)
    
    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Pass the full response to the reward function for evaluation
    reward = 1 if is_correct(response.split(), isTest=True) else -1  # Full response passed to is_correct

    return response, reward

# Load training parameters and data
with open('config.json', 'r') as f:
    config = json.load(f)
    epochs = int(config['epochs'])
    system_text = config['system']
    input_texts = config['prompts']

# Reinforcement learning fine-tuning loop
def fine_tune_with_rl(input_texts, learning_rate=5e-5, epochs=epochs):
    loss_fn = CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        total_reward = 0

        for input_text in input_texts:
            input_text = f"System: {system_text}\nUser: {input_text}"
            # Generate a response and evaluate it
            response, reward = generate_and_evaluate(input_text)
            total_reward += reward

            # Prepare the input and label tensors for loss calculation
            inputs = tokenizer(input_text, return_tensors="pt", padding=True)
            labels = tokenizer(response, return_tensors="pt", padding=True).input_ids

            # Ensure the label tensor matches the input tensor length
            labels = labels[:, :inputs.input_ids.shape[1]]  # Truncate labels to match input length

            # Move inputs and labels to the device (handled by accelerate)
            inputs, labels = accelerator.prepare(inputs, labels)

            # Forward pass
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            logits = outputs.logits

            # Compute the loss token by token
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)) * (-reward)
            total_loss += loss.item()

            # Backward pass and optimization step
            optimizer.zero_grad()
            accelerator.backward(loss)  # Use accelerator's backward pass
            optimizer.step()

        # Logging at the end of each epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}, Total Reward: {total_reward}")

# Fine-tune the model with reinforcement learning
fine_tune_with_rl(input_texts)