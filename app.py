import torch
import pandas as pd
import numpy as np
import faiss
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

# ---------------------------
# DEVICE SETUP
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- DYNAMIC PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "tinyllama-career-counselor")

# ---------------------------
# LOAD DATA
# ---------------------------
print("Loading dataset...")
df = pd.read_csv("final_20_careers_dataset.csv").fillna("Not specified")

# ---------------------------
# LOAD EMBEDDING MODEL (ON GPU)
# ---------------------------
print("Loading Embedding Model...")
embed_model = SentenceTransformer('all-mpnet-base-v2', device=device)

# ---------------------------
# PREPARE RAG INDEX
# ---------------------------
def salary_label(lpa):
    try: lpa = float(lpa)
    except: return "moderate paying"
    if lpa >= 15: return "very high paying"
    elif lpa >= 10: return "high paying"
    elif lpa >= 6: return "moderate paying"
    else: return "low paying"

def row_to_text(row):
    salary_type = salary_label(row['avg_salary_lpa'])
    return f"{row['career_name']} in {row['category']}. {salary_type} salary ({row['avg_salary_lpa']} LPA). Demand: {row['job_demand']}. Skills: {row['key_skills']}."

print("Creating embeddings...")
career_texts = df.apply(row_to_text, axis=1).tolist()
embeddings = embed_model.encode(
    career_texts,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True
)

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# ---------------------------
# LOAD FINE-TUNED MODEL (GPU)
# ---------------------------
print("Loading Fine-Tuned LLM on GPU...")

tokenizer = AutoTokenizer.from_pretrained(model_path)

llm_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,   # 🔥 faster on GPU
    device_map="auto"            # automatically places on GPU
)

llm_model.eval()

print("Backend Ready! Listening on http://localhost:8000")

# ---------------------------
# RETRIEVAL FUNCTION
# ---------------------------
def get_recommendations(query):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, 3)
    return [df.iloc[idx].to_dict() for idx in indices[0]]

# ---------------------------
# CHAT ROUTE
# ---------------------------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get("message", "")

    if not user_query:
        return jsonify({"reply": "Please enter a career-related query."})

    # RAG Retrieval
    top_careers = get_recommendations(user_query)
    context = "Top Matches:\n"

    for c in top_careers:
        context += (
            f"\nCareer: {c['career_name']}\n"
            f"Category: {c['category']}\n"
            f"Salary: {c['avg_salary_lpa']} LPA\n"
            f"Skills: {c['key_skills']}\n"
            f"Demand: {c['job_demand']}\n"
        )

    prompt = (
        f"You are an expert career counselor for Indian students.\n\n"
        f"{context}\n\n"
        f"User: {user_query}\n"
        f"Assistant:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return jsonify({"reply": response.strip()})

if __name__ == '__main__':
    app.run(port=8000)