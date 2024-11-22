from fastapi import APIRouter
from pydantic import BaseModel
from trainer import train_model
from data import generate_toy_data

router = APIRouter()

train_df, _, N_USERS, N_ITEMS = generate_toy_data()
model_instance = None


class TrainRequest(BaseModel):
    epochs: int = 3


@router.post("/")
def train(req: TrainRequest):
    global model_instance
    model_instance = train_model(train_df, N_USERS, N_ITEMS, epochs=req.epochs)
    return {"message": "Training completed and model saved", "epochs": req.epochs}
