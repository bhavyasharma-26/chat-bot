import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

import pandas as pd

df = pd.read_csv("final_20_careers_dataset.csv")

print("Shape:", df.shape)
df.head()
df.shape

df = df.fillna("Not specified") #cleaning the dataset

def salary_label(lpa):
    try:
        lpa = float(lpa)
    except:
        return "moderate paying"

    if lpa >= 15:
        return "very high paying"
    elif lpa >= 10:
        return "high paying"
    elif lpa >= 6:
        return "moderate paying"
    else:
        return "low paying"


def row_to_text(row):
    salary_type = salary_label(row['avg_salary_lpa'])

    text = (
        f"{row['career_name']} is a career in the {row['category']} domain. "
        f"It is a {salary_type} career with an average salary of {row['avg_salary_lpa']} LPA. "
        f"The job demand is {row['job_demand']} and it offers {row['global_opportunities']} global opportunities. "
        f"It requires a {row['degree_required']} degree from the {row['stream_required']} stream. "
        f"Important subjects include {row['subjects_required']}. "
        f"Key skills required are {row['key_skills']}. "
        f"The work life balance is {row['work_life_balance']}. "
        f"The difficulty level is {row['difficulty_level']} and competition level is {row['competition_level']}. "
        f"The future scope of this career is {row['future_scope']}."
    )

    return text


career_texts = df.apply(row_to_text, axis=1).tolist()

print("Total documents:", len(career_texts))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2', device=device)

import numpy as np

embeddings = model.encode(
    career_texts,
    convert_to_numpy=True,
    show_progress_bar=False
)

print("Embedding shape:", embeddings.shape)

import faiss

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("Total vectors in index:", index.ntotal)

def retrieve_careers(user_query, k=3):
    query_emb = model.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    distances, indices = index.search(query_emb, k)

    results = []

    for rank, idx in enumerate(indices[0]):
        row = df.iloc[idx]

        results.append({
            "career_name": row["career_name"],
            "similarity_score": float(distances[0][rank]),
            "category": row["category"],  # ✅ added

            "avg_salary_lpa": row["avg_salary_lpa"],
            "salary_level": salary_label(row["avg_salary_lpa"]),
            "job_demand": row["job_demand"],
            "global_opportunities": row["global_opportunities"],
            "future_scope": row["future_scope"],

            "stream_required": row["stream_required"],
            "degree_required": row["degree_required"],
            "subjects_required": row["subjects_required"],

            "key_skills": row["key_skills"],
            "difficulty_level": row["difficulty_level"],
            "competition_level": row["competition_level"],
            "work_life_balance": row["work_life_balance"]
        })

    return results

def rerank_careers(results, user_query):
    scored = []
    query_lower = user_query.lower()

    for r in results:
        score = r["similarity_score"]
        category = r["category"].lower().strip()

        # Salary Boost (stronger)
        if "high salary" in query_lower or "high paying" in query_lower:
            if r["salary_level"] in ["very high paying", "high paying"]:
                score += 0.35

        # Global Boost
        if "global" in query_lower or "international" in query_lower:
            if str(r["global_opportunities"]).lower() in ["yes", "high", "very high"]:
                score += 0.25

        # Strict Tech Filtering
        if "tech" in query_lower or "technology" in query_lower:
            if category == "tech":
                score += 0.40
            else:
                score -= 0.50   # heavy penalty

        scored.append((score, r))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [item[1] for item in scored[:3]]

query = "I want a high salary tech career with global opportunities"

results = retrieve_careers(query)

for r in results:
    print("Career:", r["career_name"])
    print("Similarity:", r["similarity_score"])
    print("Salary:", r["avg_salary_lpa"], "LPA (", r["salary_level"], ")")
    print("Stream Required:", r["stream_required"])
    print("Degree Required:", r["degree_required"])
    print("Job Demand:", r["job_demand"])
    print("Global Opportunities:", r["global_opportunities"])
    print("Future Scope:", r["future_scope"])
    print("Difficulty:", r["difficulty_level"])
    print("Competition:", r["competition_level"])
    print("Work Life Balance:", r["work_life_balance"])
    print("Skills:", r["key_skills"])
    print("--------------------------------------------------")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def build_llm_context(careers):
    context = "Top recommended careers:\n\n"

    for i, c in enumerate(careers, 1):
        context += f"""
Career {i}: {c['career_name']}
Salary: {c['avg_salary_lpa']} LPA ({c['salary_level']})
Job Demand: {c['job_demand']}
Global Opportunities: {c['global_opportunities']}
Future Scope: {c['future_scope']}
Required Stream: {c['stream_required']}
Degree Required: {c['degree_required']}
Difficulty: {c['difficulty_level']}
Competition: {c['competition_level']}
Work Life Balance: {c['work_life_balance']}
Key Skills: {c['key_skills']}
--------------------------
"""
    return context

def generate_response(user_query, careers):
    context = build_llm_context(careers)

    prompt = f"""
You are a career recommendation assistant.

User Query:
{user_query}

Here are the top matching careers retrieved from the database:

{context}

Based on the user's requirements:

1. Clearly recommend the top 3 careers.
2. For each career provide:
   - Why it matches
   - Salary and growth potential
   - Global opportunities
   - Required degree and stream
   - Key skills
   - Challenges
   - Future scope

Do not repeat the prompt.
Start directly with the recommendations.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = llm_model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.7,
        do_sample=True
    )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


initial = retrieve_careers(query, k=3)
top3 = rerank_careers(initial, query)

final_answer = generate_response(query, top3)

print(final_answer)

df[["career_name", "category"]]