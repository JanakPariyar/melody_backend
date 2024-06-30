from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Hugging Face model and tokenizer once at startup
model_name = "microsoft/DialoGPT-medium"  # Replace with your desired model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def handle_error(err):
    """Handles exceptions and returns appropriate error response."""
    print(f"Error occurred: {err}")
    return jsonify({'output': f'Error: {str(err)}'}), 500

def run(user_text, chat_history_ids=None):
    """Generates response based on user input and chat history."""
    try:
        # Encode user input with EOS token
        input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")

        # Concatenate chat history if provided
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
        else:
            bot_input_ids = input_ids

        # Generate response with appropriate settings
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Enable sampling for more diverse responses (optional)
            top_k=50,        # Consider top 50 most likely tokens at each step (optional)
            top_p=0.9,       # Filter out low probability tokens (optional)
        )

        # Decode response and strip special tokens
        resp = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return resp, chat_history_ids
    except Exception as e:
        raise e

# Create the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_text = data.get('query', '')
        print(f"Received query: {user_text}")

        # Access chat history if implemented (replace with your logic)
        chat_history_ids = None  # Implement logic to retrieve chat history token IDs

        # Generate response and update chat history
        response_text, updated_chat_history_ids = run(user_text, chat_history_ids)

        # Update chat history (replace with your logic)
        # ... (e.g., store updated_chat_history_ids in a database)

        print(f"Sending response: {response_text}")
        return jsonify({'output': response_text})

    except Exception as e:
        return handle_error(e)

if __name__ == '__main__':
    app.run(debug=True)
