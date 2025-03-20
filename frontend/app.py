import streamlit as st
import pdfplumber
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Load API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Initialize Gemini LLM via LangChain ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)

# --- Helper Function: Extract Text from PDF ---
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# --- Helper Function: Parse JSON from Response ---
def parse_json_response(response_text):
    try:
        cleaned_text = re.sub(r'[\x00-\x1F]+', '', response_text)
        json_str = re.search(r'\{.*\}', cleaned_text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception as e:
        st.error("Error parsing JSON: " + str(e))
        return None

# --- Chain for Generating Summary & Questions ---
json_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are an educational AI assistant. Given the following content, generate a concise summary and 5 educational questions based on it.\n\n"
        "Return ONLY a valid JSON object with no additional text. The JSON must have exactly two keys: 'summary' and 'questions'.\n"
        "The 'summary' value should be a string containing the summary.\n"
        "The 'questions' value should be an array of exactly 5 strings, each a question.\n\n"
        "Content: {text}"
    )
)
json_chain = LLMChain(llm=llm, prompt=json_prompt)

def generate_summary_and_questions(text):
    response_text = json_chain.run(text)
    parsed = parse_json_response(response_text)
    if parsed:
        return parsed.get("summary", ""), parsed.get("questions", [])
    else:
        st.error("Failed to parse JSON output from the AI.")
        return None, None

# --- Chain for Personalized Notes ---
notes_prompt = PromptTemplate(
    input_variables=["text", "style"],
    template=(
        "You are an educational AI assistant. Given the following content, generate concise notes in the style specified below.\n\n"
        "Style: {style}\n\n"
        "Use the following content to generate the notes:\n"
        "{text}\n\n"
        "Return ONLY the generated notes."
    )
)
notes_chain = LLMChain(llm=llm, prompt=notes_prompt)

def generate_personalized_notes(text, style):
    return notes_chain.run({"text": text, "style": style})

# --- Chain for Evaluating Answers ---
evaluation_prompt = PromptTemplate(
    input_variables=["question", "student_answer"],
    template=(
        "You are an AI teacher assistant. Evaluate the following student's answer.\n"
        "Question: {question}\n"
        "Student Answer: {student_answer}\n"
        "Give a score from 0-5 and provide feedback on improvement.\n"
        "Return a JSON object with keys 'score' and 'feedback'."
    )
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

def evaluate_answer(question, student_answer):
    response_text = evaluation_chain.run({"question": question, "student_answer": student_answer})
    return parse_json_response(response_text)

# --- Initialize session_state variables ---
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "personalized_notes" not in st.session_state:
    st.session_state["personalized_notes"] = ""
if "evaluations" not in st.session_state:
    st.session_state["evaluations"] = {}

# --- Sidebar Navigation ---
page = st.sidebar.radio("Navigation", ["Generate Content", "Questionnaire", "Dashboard"])

# --- Content Generation Page ---
if page == "Generate Content":
    st.title("📚 AI-Powered Content Generator")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    style_options = ["Story-telling", "Case Study", "Bullet Points", "Formal Summary"]
    selected_style = st.selectbox("Choose a style for personalized notes:", style_options)
    
    if uploaded_file:
        extracted_text = extract_text_from_pdf(uploaded_file)
        if st.button("Generate Content"):
            summary, questions = generate_summary_and_questions(extracted_text)
            personalized_notes = generate_personalized_notes(extracted_text, selected_style)
            if summary and questions and personalized_notes:
                st.session_state["summary"] = summary
                st.session_state["questions"] = questions
                st.session_state["personalized_notes"] = personalized_notes
                st.subheader("📌 Summary")
                st.write(summary)
                st.subheader("📝 Personalized Notes")
                st.write(personalized_notes)
                st.subheader("❓ Questions")
                for idx, q in enumerate(questions, start=1):
                    st.write(f"{idx}. {q}")

# --- Questionnaire Page ---
elif page == "Questionnaire":
    st.title("📝 Answer the Questions")
    if not st.session_state["questions"]:
        st.warning("No content generated. Please generate content first.")
    else:
        for idx, question in enumerate(st.session_state["questions"], start=1):
            st.markdown(f"**Question {idx}:** {question}")
            answer_key = f"q{idx}"
            student_answer = st.text_area(f"Your Answer for Question {idx}", key=answer_key)
            if st.button(f"Submit Answer {idx}", key=f"btn_{idx}"):
                if not student_answer.strip():
                    st.error("Please enter an answer before submitting.")
                else:
                    with st.spinner("Evaluating answer..."):
                        evaluation = evaluate_answer(question, student_answer)
                    if evaluation:
                        st.success(f"Score: {evaluation.get('score')}")
                        st.info(f"Feedback: {evaluation.get('feedback')}")
                        st.session_state["evaluations"][answer_key] = {
                            "score": evaluation.get("score"),
                            "feedback": evaluation.get("feedback")
                        }

# --- Dashboard Page ---
elif page == "Dashboard":
    st.title("📊 Performance Dashboard")
    if not st.session_state["evaluations"]:
        st.warning("No evaluations available.")
    else:
        scores = [data["score"] for data in st.session_state["evaluations"].values()]
        weak_areas = {q: d for q, d in st.session_state["evaluations"].items() if d["score"] < 3}
        
        fig, ax = plt.subplots()
        ax.pie([len(scores)-len(weak_areas), len(weak_areas)], labels=["Strong", "Weak"], autopct="%1.1f%%")
        st.pyplot(fig)

        st.subheader("📖 Suggested Learning Plan")
        if weak_areas:
            for q, data in weak_areas.items():
                st.write(f"🔴 **{q}** - {data['feedback']}")
                st.markdown("📚 Recommended Resources:")
                st.write("🔹 Revise core concepts: [Khan Academy](https://www.khanacademy.org)")
                st.write("🔹 Practice problems: [GeeksforGeeks](https://www.geeksforgeeks.org)")
        else:
            st.success("Great job! No major weak areas detected.")
