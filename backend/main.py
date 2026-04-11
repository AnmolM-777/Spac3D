import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from utils.detection import detect_objects
from utils.pipeline import (
    extract_features,
    reduce_features,
    classify_style,
    recommend_furniture,
    _get_resnet,
    _load_sklearn_models,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== SPAC3D Backend Starting Up ===")
    _get_resnet()
    _load_sklearn_models()
    logger.info("=== All models ready. Server is live. ===")
    yield
    logger.info("=== SPAC3D Backend Shutting Down ===")


app = FastAPI(
    title="SPAC3D API",
    description="Room analysis and furniture recommendation using a multi-model ML pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SPAC3D API is running"}


@app.post("/analyze")
async def analyze_room(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        content = await image.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        # Step 1: Object Detection (YOLOv8)
        logger.info("Step 1/5: YOLOv8 object detection...")
        detections, person_count, total_boxes = detect_objects(pil_image)

        # Step 2: Feature Extraction (ResNet50 — Transfer Learning)
        logger.info("Step 2/5: ResNet50 feature extraction...")
        embedding_2048 = extract_features(pil_image)

        # Step 3: Dimensionality Reduction (PCA)
        logger.info("Step 3/5: PCA reduction (2048 -> 32 dims)...")
        embedding_32 = reduce_features(embedding_2048)

        # Step 4: Style Classification (kNN + SVM)
        logger.info("Step 4/5: kNN + SVM style classification...")
        style, confidence = classify_style(embedding_32)

        # Room Validation Heuristic: Reject if it's unlikely to be a room
        is_group_photo = person_count > 0 and len(detections) == 0
        is_random_photo = len(detections) == 0 and confidence < 0.40
        
        if is_group_photo or is_random_photo:
            logger.warning(f"Rejected as non-room: furniture={len(detections)}, persons={person_count}, confidence={confidence:.2f}")
            raise HTTPException(
                status_code=400,
                detail="This doesn't look like a room. Please upload a clear photo of an indoor space (like a living room, bedroom, etc). We couldn't detect any furniture or recognize the room style."
            )

        # Step 5: Furniture Recommendation (Cosine Similarity)
        # We request top_n=100 to get the full catalog (which has ~42 items)
        # This allows the frontend to instantly filter recommendations by category when a bounding box is clicked.
        logger.info("Step 5/5: Cosine similarity furniture recommendation...")
        recommendations = recommend_furniture(style, embedding_32, top_n=100)

        for rec in recommendations:
            rec.pop("feature_vector", None)

        logger.info(f"Analysis complete: style={style}, confidence={confidence:.2%}, detections={len(detections)}")

        return {
            "status": "success",
            "detections": detections,
            "style": style,
            "confidence": confidence,
            "recommendations": recommendations,
            "image_size": {"width": pil_image.width, "height": pil_image.height},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

