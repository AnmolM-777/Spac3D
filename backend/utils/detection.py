from ultralytics import YOLOWorld
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

INTERIOR_CLASSES = [
    "person", "chair", "couch", "sofa", "bed", "dining table", "table",
    "potted plant", "plant", "tv", "television", "refrigerator",
    "clock", "vase", "book", "lamp", "laptop", "monitor",
    "bench", "desk", "cabinet", "cushion", "pillow", "shelf",
    "drawer", "rug", "mirror", "painting", "ottoman", "stool",
    "sideboard", "console"
]

_yolo_model = None

def get_yolo_model() -> YOLOWorld:
    global _yolo_model
    if _yolo_model is None:
        logger.info("Loading YOLO-World open-vocabulary model...")
        _yolo_model = YOLOWorld("yolov8s-world.pt")
        _yolo_model.set_classes(INTERIOR_CLASSES)
        logger.info("YOLO-World loaded with custom vocabulary.")
    return _yolo_model

def detect_objects(pil_image: Image.Image) -> tuple[list[dict], int, int]:
    model = get_yolo_model()
    # Using 'cpu' to avoid CUDA memory conflicts from previous issues
    results = model(pil_image, conf=0.15, verbose=False, device='cpu')

    relevant_categories = set(INTERIOR_CLASSES) - {"person"}

    detections = []
    person_count = 0
    total_boxes = 0
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            total_boxes += 1
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # YOLO-World returns dynamic class IDs based on our set_classes
            if cls_id < len(INTERIOR_CLASSES):
                class_name = INTERIOR_CLASSES[cls_id]
            else:
                class_name = "unknown"
            
            if class_name.lower() == "person":
                person_count += 1
                
            if class_name.lower() not in relevant_categories:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": class_name,
                "confidence": round(conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            })

    # Custom Cross-Class Non-Maximum Suppression (via Intersection over smaller Area)
    # This prevents YOLO-World from tagging a section of a "sofa" as a separate "bench"
    def compute_ioa(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        minArea = min(boxAArea, boxBArea)
        if minArea == 0:
            return 0
        return interArea / float(minArea)

    MACRO_CLASSES = {"chair", "couch", "sofa", "bed", "dining table", "table", "bench", "desk", "cabinet", "ottoman", "stool", "sideboard", "console"}
    
    # Sort detections by confidence descending
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    
    filtered_detections = []
    for d in detections:
        keep = True
        is_macro = d["label"] in MACRO_CLASSES
        for fd in filtered_detections:
            # Synonyms suppression (if two identical boxes are called cushion and pillow)
            is_synonym = (d["label"] in {"cushion", "pillow"} and fd["label"] in {"cushion", "pillow"}) or \
                         (d["label"] in {"sofa", "couch"} and fd["label"] in {"sofa", "couch"})
            
            if is_macro and fd["label"] in MACRO_CLASSES:
                if compute_ioa(d["bbox"], fd["bbox"]) > 0.65:
                    keep = False
                    break
            elif is_synonym:
                if compute_ioa(d["bbox"], fd["bbox"]) > 0.8:
                    keep = False
                    break
        if keep:
            filtered_detections.append(d)

    logger.info(f"YOLOv8 detected {len(filtered_detections)} furniture items, {person_count} people, {total_boxes} total objects.")
    return filtered_detections, person_count, total_boxes
