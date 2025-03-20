import streamlit as st
import pdfplumber
import os
import json
import re
import tempfile
import subprocess
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# (Assuming you have installed and set up all necessary packages)

# ---------- Load API Key ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------- Initialize Gemini LLM via LangChain ----------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)

# ---------- Helper Function: Extract Text from PDF ----------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---------- Helper Function: Parse JSON from Response ----------
def parse_json_response(response_text):
    try:
        cleaned_text = re.sub(r'[\x00-\x1F]+', '', response_text)
        json_str = re.search(r'\{.*\}', cleaned_text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception as e:
        st.error("Error parsing JSON: " + str(e))
        return None

# ---------- Combined Chain for Generating Personalized Notes and Questions ----------
combined_prompt = PromptTemplate(
    input_variables=["text", "style"],
    template=(
        "You are an educational AI assistant. Given the following content, generate personalized notes in the style specified and 5 educational questions based on it.\n\n"
        "Style: {style}\n\n"
        "Return ONLY a valid JSON object with no additional text. The JSON must have exactly two keys: 'notes' and 'questions'.\n"
        "The 'notes' value should be a string containing the personalized notes in the given style.\n"
        "The 'questions' value should be an array of exactly 5 strings, each a question.\n\n"
        "Content: {text}"
    )
)
combined_chain = LLMChain(llm=llm, prompt=combined_prompt)

def generate_content(text, style):
    response_text = combined_chain.run({"text": text, "style": style})
    parsed = parse_json_response(response_text)
    if parsed:
        notes = parsed.get("notes", "")
        questions = parsed.get("questions", [])
        return notes, questions
    else:
        st.error("Failed to parse JSON output from the AI. Please try again.")
        return None, None

# ---------- Chain for Flashcards Generation ----------
flashcard_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are an educational AI assistant. Based on the following content, generate a set of flashcards to help review key concepts. "
        "Return ONLY a valid JSON object with one key 'flashcards'. The value should be an array of objects, each with keys 'question' and 'answer'.\n\n"
        "Content: {text}"
    )
)
flashcard_chain = LLMChain(llm=llm, prompt=flashcard_prompt)

def generate_flashcards(text):
    response_text = flashcard_chain.run(text)
    parsed = parse_json_response(response_text)
    if parsed and "flashcards" in parsed:
        return parsed["flashcards"]
    else:
        st.error("Failed to generate flashcards.")
        return None

# ---------- Chain for Evaluating Answers ----------
evaluation_prompt = PromptTemplate(
    input_variables=["question", "student_answer"],
    template=(
        "You are an educational AI assistant. Evaluate the student's answer for the following question objectively.\n"
        "Question: {question}\n"
        "Student Answer: {student_answer}\n"
        "Also detect if the answer submitted by the student was AI generated; if yes, give a score of -1 with feedback as 'AI Generated Answer', otherwise, "
        "provide a score between 0 and 5 with detailed feedback on how to improve the answer.\n"
        "Return ONLY a valid JSON object with keys 'score' and 'feedback'."
    )
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

def evaluate_answer(question, student_answer):
    response_text = evaluation_chain.run({"question": question, "student_answer": student_answer})
    return parse_json_response(response_text)

# ---------- New Chain for Generating PlantUML Code (Visual Insights) ----------
plantuml_prompt = PromptTemplate(
    input_variables=["notes"],
    template=(
        "You are an expert in UML diagramming. Given the following educational content, "
        "generate a diagram in standard PlantUML syntax. Do not include any external references or libraries. "
        "Return ONLY the PlantUML code.\n\n"
        "Content: {notes}"
    )
)
plantuml_chain = LLMChain(llm=llm, prompt=plantuml_prompt)

def generate_plantuml_code(notes):
    return plantuml_chain.run({"notes": notes})

def generate_uml(uml_code, output_filename="diagram.png"):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.uml', delete=False) as temp_file:
        uml_filepath = temp_file.name
        temp_file.write(uml_code)
    
    # Specify the full path to the PlantUML executable
    plantuml_executable = r"C:\ProgramData\chocolatey\bin\plantuml.cmd"
    command = [plantuml_executable, '-tpng', uml_filepath]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        st.error("Error generating UML diagram: " + result.stderr.decode())
        os.remove(uml_filepath)
        return None
    
    generated_filepath = uml_filepath.replace('.uml', '.png')
    if os.path.exists(generated_filepath):
        os.rename(generated_filepath, output_filename)
        os.remove(uml_filepath)
        return output_filename
    else:
        st.error("Diagram file not found.")
        os.remove(uml_filepath)
        return None


# ---------- New Chain for Lesson Planning ----------
lesson_planning_prompt = PromptTemplate(
    input_variables=["plan_type", "subject", "grade_level", "objectives", "num_days"],
    template=(
        "You are an AI that assists teachers in creating lesson materials.\n\n"
        "Plan Type: {plan_type}\n"
        "Subject: {subject}\n"
        "Grade Level: {grade_level}\n"
        "Objectives: {objectives}\n"
        "Number of Days: {num_days}\n\n"
        "Instructions:\n"
        "- If the plan type is 'Lesson Seed', provide a brief outline that the teacher can expand.\n"
        "- If the plan type is 'Lesson Plan', provide a detailed lesson structure (objectives, activities, assessments).\n"
        "- If the plan type is 'Unit Plan', outline a multi-week approach with subtopics and key activities.\n"
        "- If the plan type is 'Plan by Number of Days', break the plan into day-by-day sections.\n\n"
        "Return only the plan details with minimal additional text."
    )
)
lesson_planning_chain = LLMChain(llm=llm, prompt=lesson_planning_prompt)

def generate_lesson_plan(plan_type, subject, grade_level, objectives, num_days):
    response = lesson_planning_chain.run({
        "plan_type": plan_type,
        "subject": subject,
        "grade_level": grade_level,
        "objectives": objectives,
        "num_days": num_days
    })
    return response

# ---------- Initialize session_state Variables ----------
if "notes" not in st.session_state:
    st.session_state["notes"] = ""
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "evaluations" not in st.session_state:
    st.session_state["evaluations"] = {}
if "plantuml_code" not in st.session_state:
    st.session_state["plantuml_code"] = ""
if "diagram_path" not in st.session_state:
    st.session_state["diagram_path"] = ""
if "show_uml_code" not in st.session_state:
    st.session_state["show_uml_code"] = False
if "lesson_plan" not in st.session_state:
    st.session_state["lesson_plan"] = ""
if "flashcards" not in st.session_state:
    st.session_state["flashcards"] = []

# ---------- Sidebar Navigation ----------
page = st.sidebar.radio("Navigation", [
    "Generate Content",
    "Flashcards",
    "Questionnaire",
    "Dashboard",
    "Visual Insights",
    "Lesson Planning"
])

# ---------- Page 1: Generate Content (Notes, Summary & Questions) ----------
if page == "Generate Content":
    st.title("📚 Generate Content")
    st.subheader("Upload a PDF or Notes File")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    style_options = ["Story-telling", "Case Study", "Bullet Points", "Formal Summary", "As complete explanation"]
    selected_style = st.selectbox("Choose a style for personalized notes:", style_options)
    
    if uploaded_file:
        with st.spinner("Extracting text..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        if extracted_text:
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text (Preview)", extracted_text[:1000], height=200)
            if st.button("Generate Content"):
                with st.spinner("Generating personalized notes and questions..."):
                    notes, questions = generate_content(extracted_text, selected_style)
                if notes and questions:
                    st.session_state["notes"] = notes
                    st.session_state["questions"] = questions
                    st.subheader("📝 AI-Generated Personalized Notes")
                    st.write(notes)

# ---------- Page 2: Flashcards ----------
elif page == "Flashcards":
    st.title("🃏 Flashcards")
    if not st.session_state["notes"]:
        st.warning("No content generated. Please generate content first.")
    else:
        st.subheader("Flashcards Generated from Content")
        flashcards = generate_flashcards(st.session_state["notes"])
        if flashcards:
            st.session_state["flashcards"] = flashcards
            for idx, card in enumerate(flashcards, start=1):
                with st.expander(f"Flashcard {idx}: {card.get('question', 'No Question Provided')}"):
                    st.write("**Answer:**", card.get("answer", "No Answer Provided"))
        else:
            st.error("Flashcards could not be generated.")

# ---------- Page 3: Questionnaire and Evaluation ----------
elif page == "Questionnaire":
    st.title("📝 Questionnaire")
    if not st.session_state["questions"]:
        st.warning("No content generated. Please generate content first.")
    else:
        st.subheader("Answer the Following Questions")
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
        with st.expander("Show All Evaluations"):
            if st.session_state["evaluations"]:
                for key, eval_data in st.session_state["evaluations"].items():
                    st.write(f"**{key}:** Score: {eval_data.get('score')}, Feedback: {eval_data.get('feedback')}")
            else:
                st.info("No evaluations available.")

# ---------- Page 4: Dashboard (Visual Insights & Performance) ----------
elif page == "Dashboard":
    st.title("📊 Performance Dashboard")
    if not st.session_state["evaluations"]:
        st.warning("No evaluations available. Please complete the questionnaire first.")
    else:
        scores = [data["score"] for data in st.session_state["evaluations"].values()]
        weak_areas = {q: d for q, d in st.session_state["evaluations"].items() if d["score"] < 3}
        
        fig, ax = plt.subplots()
        ax.pie([len(scores) - len(weak_areas), len(weak_areas)], 
               labels=["Strong", "Weak"], autopct="%1.1f%%", colors=["#4CAF50", "#FF5733"],
               startangle=90)
        ax.axis("equal")
        st.subheader("📌 Performance Breakdown")
        st.pyplot(fig)
        
        st.subheader("📖 Suggested Learning Plan")
        if weak_areas:
            for q, data in weak_areas.items():
                st.markdown(f"🔴 **{q}**")
                st.write(f"Feedback: {data['feedback']}")
                st.markdown("**Recommended Resources:**")
                st.write("🔹 Revise core concepts: [Khan Academy](https://www.khanacademy.org)")
                st.write("🔹 Practice problems: [GeeksforGeeks](https://www.geeksforgeeks.org)")
        else:
            st.success("Great job! No major weak areas detected.")

# ---------- Page 5: Visual Insights (UML Diagram) ----------
elif page == "Visual Insights":
    st.title("🎨 Visual Insights")
    if not st.session_state["notes"]:
        st.warning("No content generated. Please generate content first to create a diagram.")
    else:
        if not st.session_state["plantuml_code"]:
            with st.spinner("Generating PlantUML code from content..."):
                plantuml_code = generate_plantuml_code(st.session_state["notes"])
                st.session_state["plantuml_code"] = plantuml_code
        else:
            plantuml_code = st.session_state["plantuml_code"]
        
        if not st.session_state["diagram_path"]:
            with st.spinner("Generating diagram using PlantUML..."):
                diagram_path = generate_uml(plantuml_code, output_filename="diagram.png")
                if diagram_path:
                    st.session_state["diagram_path"] = diagram_path
        
        if st.session_state["diagram_path"]:
            st.image(st.session_state["diagram_path"], caption="Generated Diagram", use_column_width=True)
            st.success("Diagram generated successfully.")
        
        if st.button("Show/Hide UML Code"):
            st.session_state["show_uml_code"] = not st.session_state["show_uml_code"]
        
        if st.session_state["show_uml_code"]:
            st.code(st.session_state["plantuml_code"], language="plantuml")

# ---------- Page 6: Lesson Planning ----------
elif page == "Lesson Planning":
    st.title("📅 Lesson Planning")
    st.subheader("Create AI-generated lesson materials")
    
    plan_type = st.radio(
        "Select Plan Type",
        ["Lesson Seed", "Lesson Plan", "Unit Plan", "Plan by Number of Days"]
    )
    subject = st.text_input("Subject (e.g., Math, English)")
    grade_level = st.text_input("Grade Level (e.g., Grade 5)")
    objectives = st.text_area("Learning Objectives (optional)")
    if plan_type == "Plan by Number of Days":
        num_days = st.number_input("Number of Days", min_value=1, max_value=10, value=5)
    else:
        num_days = 1
    
    if st.button("Generate Lesson Plan"):
        with st.spinner("Generating lesson plan..."):
            lesson_text = generate_lesson_plan(
                plan_type,
                subject,
                grade_level,
                objectives,
                num_days
            )
        st.subheader("Your AI-Generated Plan")
        st.write(lesson_text)
        st.session_state["lesson_plan"] = lesson_text
