from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Chat history
chat_history_ids = None

print("🤖 Chatbot is ready! Type 'exit' to quit.\n")

for step in range(10):  # Max 10 interactions
    user_input = input("👤 You: ")
    if user_input.lower() == "exit":
        break

    # Encode user input and add to chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate a response
    chat_history_ids = model.generate(bot_input_ids, max_length=100000, pad_token_id=tokenizer.eos_token_id)

    # Decode the last output token
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    print(f"🤖 Bot: {response}")
