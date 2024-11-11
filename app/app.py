import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.load_model import pipe


class UserInput(BaseModel):
    user_input: str

class ModelOutput(BaseModel):
    label: str
    score: float


description = """
API для оценки саркастичности текста
"""

logging.info('Starting FastAPI')
app = FastAPI(
    title="Sarcasm-analysis",
    description=description,
    summary='Анализ саркастичности',
    version="0.0.1"
)

@app.post("/sarcasm", response_model=ModelOutput)
async def predict_answer(user_input: UserInput):
    try:
        text = user_input.user_input
        output = pipe(text)
        return ModelOutput(label=output[0]['label'], score=output[0]['score'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))