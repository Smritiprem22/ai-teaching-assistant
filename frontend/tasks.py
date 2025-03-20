from celery import Celery
import os
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API Key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Celery (using Redis as broker; adjust the URL as needed)
app = Celery('tasks', broker='redis://localhost:6379/0')

# Initialize Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY)

# Define the evaluation prompt
evaluation_prompt = PromptTemplate(
    input_variables=["question", "student_answer"],
    template=(
        "You are an educational AI assistant. Evaluate the student's answer for the following question:\n"
        "Question: {question}\n"
        "Student Answer: {student_answer}\n"
        "Provide a score between 1 and 10 and detailed feedback on how to improve the answer.\n"
        "Return ONLY a valid JSON object with keys 'score' and 'feedback'."
    )
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

def parse_json_response(response_text):
    try:
        json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
        return json.loads(json_str)
    except Exception:
        return None

@app.task
def evaluate_answer_task(question, student_answer):
    # Run the evaluation chain
    response_text = evaluation_chain.run({"question": question, "student_answer": student_answer})
    evaluation_data = parse_json_response(response_text)
    return evaluation_data
