from fastapi import APIRouter, HTTPException
import torch
from model import MFModel
from data import generate_toy_data
import os

router = APIRouter()

train_df, _, N_USERS, N_ITEMS = generate_toy_data()
MODEL_PATH = "models/model_store/model.pt"
model_instance = None


def load_model():
    global model_instance
    if not os.path.exists(MODEL_PATH):
        print("No model found. Train first.")
        return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = MFModel(checkpoint["n_users"], checkpoint["n_items"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Model loaded successfully.")
    return model


@router.get("/{user_id}")
def recommend(user_id: int, k: int = 10):
    global model_instance
    if model_instance is None:
        model_instance = load_model()
        # raise HTTPException(status_code=400, detail="Model not trained. POST /train first.")
    if not (0 <= user_id < N_USERS):
        raise HTTPException(status_code=404, detail="Invalid user ID")

    with torch.no_grad():
        user_vec = torch.tensor([user_id] * N_ITEMS)
        item_vec = torch.tensor(range(N_ITEMS))
        scores = model_instance(user_vec, item_vec).numpy()

    seen = set(train_df[train_df["user"] == user_id]["item"].tolist())
    candidates = [(i, float(scores[i])) for i in range(N_ITEMS) if i not in seen]
    candidates.sort(key=lambda x: x[1], reverse=True)

    return {"recommendations": candidates[:k]}
