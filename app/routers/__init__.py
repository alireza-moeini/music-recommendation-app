from fastapi import APIRouter
from .train import router as train_router
from .recommend import router as recommend_router

router = APIRouter()
router.include_router(train_router, prefix="/train", tags=["Training"])
router.include_router(recommend_router, prefix="/recommend", tags=["Recommendation"])
