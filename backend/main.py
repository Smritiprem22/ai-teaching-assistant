from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Teaching Assistant API!"}

# Run the server using: uvicorn main:app --reload
