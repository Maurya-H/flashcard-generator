
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

app = FastAPI(
    title="Flashcard Generator API",
    description="Generate educational flashcards using Hugging Face",
    version="1.0.0"
)

client = InferenceClient(
    provider="auto",
    api_key=token
) if token else None


class FlashcardRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    number_of_cards: int = Field(default=5, ge=1, le=20)
    difficulty: str = Field(default="Beginner")


@app.get("/")
def home():
    return {
        "message": "Flashcard Generator API is running",
        "docs": "/docs"
    }


@app.post("/generate-flashcards")
def generate_flashcards(request: FlashcardRequest):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is missing. Check your .env file."
        )

    prompt = f"""
    Create exactly {request.number_of_cards} educational flashcards
    about {request.topic}.
    Difficulty: {request.difficulty}

    Format each card as:
    Question: ...
    Answer: ...

    Number every card. Keep answers clear and accurate.
    """

    try:
        print(f"\nGenerating flashcards about: {request.topic}")

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200
        )

        answer = response.choices[0].message.content

        print("\n===== GENERATED FLASHCARDS =====")
        print(answer)
        print("================================\n")

        return {
            "topic": request.topic,
            "number_of_cards": request.number_of_cards,
            "difficulty": request.difficulty,
            "flashcards": answer
        }

    except Exception as e:
        error = str(e)
        print(f"\nHugging Face error: {error}")

        if "402" in error or "credits" in error.lower():
            detail = (
                "Hugging Face inference credits are exhausted. "
                "Changing models does not necessarily fix this. "
                "Check your Inference Providers billing and credits."
            )
            raise HTTPException(status_code=402, detail=detail)

        raise HTTPException(
            status_code=502,
            detail=f"Hugging Face generation failed: {error}"
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)