import os
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Load API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Initialize the client
client = genai.Client(api_key=api_key)

@app.route("/")
def home():
    return "Welcome to Character Chat API!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    character_name = data.get("character", "Unknown Character")
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # System prompt
    system_prompt = (
        f"You are {character_name}. Respond as they would."
    )

    conversation_history = [
        types.Content(role="user", parts=[types.Part(text=system_prompt)]),
        types.Content(role="user", parts=[types.Part(text=user_input)])
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=conversation_history,
            config=types.GenerateContentConfig(
                temperature=0.9, top_p=0.95, top_k=40, max_output_tokens=8192
            )
        )
        
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
