import os
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from groq import Groq

load_dotenv()

app = Flask(__name__)

# ------------------------------------------------------------
# Basic preprocessing
# ------------------------------------------------------------
def preprocess(text: str) -> dict:
    original = text.strip()
    lowered = original.lower()
    processed = re.sub(r"[^\w\s]", " ", lowered)
    processed = re.sub(r"\s+", " ", processed).strip()
    tokens = processed.split()
    return {"original": original, "processed": processed, "tokens": tokens}


# ------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------
def build_prompt(preprocessed: dict) -> str:
    prompt = (
        "You are an assistant that answers user questions concisely and accurately.\n\n"
        f"User question (original): {preprocessed['original']}\n"
        f"User question (processed): {preprocessed['processed']}\n\n"
        "Answer the user's question directly. If the question is ambiguous, ask one brief clarifying "
        "question. Keep the answer clear and show one short paragraph. If you must give step-by-step, "
        "number them.\n\nAnswer:"
    )
    return prompt


# ------------------------------------------------------------
# Query Groq API
# ------------------------------------------------------------
def query_groq(prompt: str, max_tokens=400):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return "Error: GROQ_API_KEY not found. Please create a .env file with your key."

    client = Groq(api_key=key)

    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq API error: {e}"


# ------------------------------------------------------------
# LLM router
# ------------------------------------------------------------
def send_to_llm(prompt: str):
    if os.getenv("GROQ_API_KEY"):
        return query_groq(prompt)
    return (
        "No LLM backend configured.\n"
        "Here is your processed prompt instead:\n\n"
        f"{prompt}"
    )


# ------------------------------------------------------------
# Flask routes
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question", "").strip()

    if not user_question:
        return jsonify({"error": "No question provided."})

    pre = preprocess(user_question)
    prompt = build_prompt(pre)
    answer = send_to_llm(prompt)

    return jsonify({
        "processed": pre["processed"],
        "tokens": pre["tokens"],
        "answer": answer
    })


if __name__ == "__main__":
    app.run(debug=True)
