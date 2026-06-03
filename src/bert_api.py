from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict_fun import Predictor


app = FastAPI(title="Resume Classification BERT API")
predictor = Predictor()


class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: int
    category: str
    time_ms: float


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TextRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    try:
        return predictor.predict_single(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def read_root():
    return {"message": "BERT resume classification API is running!"}
