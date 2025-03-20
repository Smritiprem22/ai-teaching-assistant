import streamlit as st
import pdfplumber
import os
import json
import re
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
        # Clean up control characters
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
        summary = parsed.get("summary", "")
        questions = parsed.get("questions", [])
        return summary, questions
    else:
        st.error("Failed to parse JSON output from the AI. Please try again.")
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

# --- Chain for Objective Evaluation of Answers ---
evaluation_prompt = PromptTemplate(
    input_variables=["question", "student_answer"],
    template=(
        "You are an educational AI assistant. Evaluate the student's answer for the following question objectively.\n"
        "Question: {question}\n"
        "Student Answer: {student_answer}\n"
        "Provide a score between 0 and 5 and detailed feedback on how to improve the answer.\n"
        "Return ONLY a valid JSON object with keys 'score' and 'feedback'."
    )
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

def evaluate_answer(question, student_answer):
    response_text = evaluation_chain.run({"question": question, "student_answer": student_answer})
    evaluation = parse_json_response(response_text)
    if evaluation:
        return evaluation
    else:
        st.error("Evaluation failed to produce valid JSON.")
        return None

# --- Initialize session_state variables if not already set ---
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "personalized_notes" not in st.session_state:
    st.session_state["personalized_notes"] = ""
if "evaluations" not in st.session_state:
    st.session_state["evaluations"] = {}

# --- Navigation: Sidebar for Multi-Page Layout ---
page = st.sidebar.radio("Navigation", ["Generate Content", "Questionnaire"])

if page == "Generate Content":
    st.title("📚 Generate Content")
    st.subheader("Upload a PDF or Notes File")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    # Section for personalized notes style
    style_options = ["Story-telling", "Case Study", "Bullet Points", "Formal Summary"]
    selected_style = st.selectbox("Choose a style for personalized notes:", style_options)
    
    if uploaded_file:
        with st.spinner("Extracting text..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        if extracted_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", extracted_text[:1000], height=200)
            if st.button("Generate Content"):
                with st.spinner("Generating summary, questions, and personalized notes..."):
                    summary, questions = generate_summary_and_questions(extracted_text)
                    personalized_notes = generate_personalized_notes(extracted_text, selected_style)
                if summary and questions and personalized_notes:
                    st.session_state["summary"] = summary
                    st.session_state["questions"] = questions
                    st.session_state["personalized_notes"] = personalized_notes
                    
                    st.subheader("📌 AI-Generated Summary")
                    st.write(summary)
                    st.subheader("📝 Personalized Notes")
                    st.write(personalized_notes)
                    st.subheader("❓ AI-Generated Questions")
                    for idx, q in enumerate(questions, start=1):
                        st.markdown(f"*Question {idx}:* {q}")

elif page == "Questionnaire":
    st.title("📝 Questionnaire")
    if not st.session_state["questions"]:
        st.warning("No content generated. Please generate content first.")
    else:
        st.subheader("Review Summary")
        st.write(st.session_state["summary"])
        st.subheader("Review Personalized Notes")
        st.write(st.session_state["personalized_notes"])
        st.subheader("Answer the Following Questions")
        # For each question, collect answer and then evaluate it on submission.
        for idx, question in enumerate(st.session_state["questions"], start=1):
            st.markdown(f"*Question {idx}:* {question}")
            answer_key = f"q{idx}"
            # Provide a text area for student's answer.
            student_answer = st.text_area(f"Your Answer for Question {idx}", key=answer_key)
            if st.button(f"Submit Answer for Question {idx}", key=f"btn_{idx}"):
                if not student_answer.strip():
                    st.error("Please enter an answer before submitting.")
                else:
                    with st.spinner("Evaluating answer..."):
                        evaluation = evaluate_answer(question, student_answer)
                    if evaluation:
                        # Only display grade and feedback.
                        st.success(f"Score: {evaluation.get('score')}")
                        st.info(f"Feedback: {evaluation.get('feedback')}")
                        # Store only the evaluation results.
                        st.session_state["evaluations"][answer_key] = {
                            "score": evaluation.get("score"),
                            "feedback": evaluation.get("feedback")
                        }
        if st.button("Show All Evaluations"):
            st.subheader("Evaluations Summary")
            # Create a clean table of evaluations.
            if st.session_state["evaluations"]:
                for key, eval_data in st.session_state["evaluations"].items():
                    st.write(f"{key}:** Score: {eval_data.get('score')}, Feedback: {eval_data.get('feedback')}")
            else:
                st.info("No evaluations available.")