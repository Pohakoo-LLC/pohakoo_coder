import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from validator import is_correct
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load pre-trained Phi 3 mini 4k model and tokenizer
model_name = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token to be the same as EOS token if needed
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Function to generate responses and calculate rewards
def generate_and_evaluate(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    
    # Generate text using the model
    outputs = model.generate(inputs.input_ids, max_length=256, attention_mask=inputs.attention_mask)
    
    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Pass the full response to the reward function for evaluation
    reward = 1 if is_correct(response.split()) else -1  # Full response passed to is_correct

    return response, reward

with open('config.json', 'r') as f:
    epochs = json.load(f)['epochs']
    epochs = int(epochs)

# Reinforcement learning fine-tuning loop
def fine_tune_with_rl(input_texts, learning_rate=5e-5, epochs=epochs):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()
    with open('config.json', 'r') as f:
        system_text = json.load(f)['system']

    for epoch in range(epochs):
        total_loss = 0
        total_reward = 0
        i = 0

        for input_text in input_texts:
            input_text = f"System: {system_text}\nUser: {input_text}"
            # Generate a response and evaluate it
            response, reward = generate_and_evaluate(input_text)
            total_reward += reward

            # Prepare the input and label tensors for loss calculation
            inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
            labels = tokenizer(response, return_tensors="pt", padding=True).input_ids.to(device)

            # Ensure the label tensor matches the input tensor length
            labels = labels[:, :inputs.input_ids.shape[1]]  # Truncate labels to match input length

            # Forward pass
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            logits = outputs.logits

            # Compute the loss token by token
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)) * (-reward)
            total_loss += loss.item()

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Display percent completion
            i+=1
            percent_complete = i / len(input_texts) * 100
            logging.info(f"Epoch {epoch}/{epochs}: {percent_complete:.2f}% complete")

        # Logging at the end of each epoch
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}, Total Reward: {total_reward}")

# Sample training data (list of prompts)
with open('prompts.txt', 'r') as f:
    input_texts = f.readlines()

# Fine-tune the model with reinforcement learning
fine_tune_with_rl(input_texts)