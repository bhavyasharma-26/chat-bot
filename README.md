# Fine-Tuned Career Counselor Bot

An AI-based career counseling chatbot designed to help students explore suitable career options based on their interests, skills, salary expectations, and other career-related factors.

The project combines a fine-tuned TinyLlama language model with a Retrieval-Augmented Generation (RAG) system to provide career-related responses using a custom career dataset.

## About the Project

Choosing a career can be difficult for students because there are many career options and different factors to consider.

This project aims to provide a career counseling assistant that can:

- Answer career-related questions
- Recommend relevant career options
- Provide salary information
- Show required skills
- Provide information about job demand
- Use career data to support its responses
- Generate personalized responses using a fine-tuned language model

## Features

### Fine-Tuned Language Model

TinyLlama-1.1B-Chat-v1.0 was fine-tuned using LoRA on a custom career counseling dataset.

### Retrieval-Augmented Generation

The system uses FAISS and Sentence Transformers to retrieve the most relevant career information from the career dataset before generating a response.

### Career Dataset

The career dataset contains information such as:

- Career name
- Career category
- Average salary
- Job demand
- Required skills
- Educational requirements
- Subjects
- Global opportunities
- Future scope
- Difficulty level
- Competition level
- Work-life balance

### Web Interface

A simple web-based interface allows users to interact with the Career Counselor Bot.

## Project Workflow

```text
User Query
    ↓
Sentence Transformer
    ↓
FAISS Similarity Search
    ↓
Top 3 Relevant Careers
    ↓
Career Information + User Query
    ↓
TinyLlama + LoRA Adapter
    ↓
Generated Career Counseling Response
