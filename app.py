from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image
import uvicorn
import os
import pathlib
import torch
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

# ====================================================
# 1. Pydantic 모델 정의 (API 명세 강화)
# ====================================================
class DetectionResult(BaseModel):
    class_name: str = Field(..., description="탐지된 객체의 클래스 이름")
    confidence: float = Field(..., description="탐지에 대한 신뢰도 점수 (0.0 ~ 1.0)")
    box: List[int] = Field(..., description="[x1, y1, x2, y2] 형식의 바운딩 박스 좌표")

class PredictionResponse(BaseModel):
    status: str = Field(..., description="탐지 결과 상태 ('item_detected' 또는 'no_item_detected')")
    best_detection: Optional[DetectionResult] = Field(None, description="가장 신뢰도 높은 탐지 결과 (하나)")

# ====================================================
# 2. 환경 및 경로 설정
# ====================================================
# app.py 파일의 위치를 기준으로 상대 경로를 설정합니다.
# Dockerfile에서 `COPY . .`를 실행하면, `ai` 폴더와 `app.py`가 같은 `/app` 디렉토리에 위치하게 됩니다.
MODEL_WEIGHTS_PATH = pathlib.Path(__file__).parent / "model" / "best.pt"

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ====================================================
# 3. FastAPI 앱 초기화 및 모델 로딩
# ====================================================
app = FastAPI(
    title="재활용품 분류 YOLOv8 API",
    description="이미지 속 재활용품 중 가장 정확도가 높은 객체 하나를 탐지합니다.",
    version="1.1.0"
)

# CORS 미들웨어 추가
# 프론트엔드 애플리케이션(예: http://localhost:3000)에서 오는 요청을 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 프로덕션에서는 ["http://your-frontend-domain.com"] 와 같이 특정 도메인을 지정하는 것이 안전합니다.
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)


model = None
 
@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델을 전역 변수로 한 번만 로드합니다."""
    global model
    if not MODEL_WEIGHTS_PATH.exists():
        # 모델 파일이 없으면 서버 실행을 중단시키는 것이 더 안전합니다.
        raise RuntimeError(f"FATAL: Model file not found at {MODEL_WEIGHTS_PATH}")

    try:
        model = YOLO(MODEL_WEIGHTS_PATH)
        model.to(DEVICE) # 모델을 지정된 장치로 이동
        print("✅ YOLOv8 Model Loaded Successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Model Loading Error: {e}")

# ====================================================
# 4. API 엔드포인트
# ====================================================
@app.post("/predict/recycle_item", response_model=PredictionResponse, summary="재활용품 탐지 (Best 1)")
async def predict_image(file: UploadFile = File(...)):
    """
    이미지를 받아 가장 신뢰도 높은 재활용품 1개를 탐지하여 반환합니다.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialization failed.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    try:
        # 1. 이미지 읽기 및 변환
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. YOLO 모델 예측 수행
        results = model.predict(source=image, conf=0.6, iou=0.5, device=DEVICE, verbose=False)

        # 3. 결과 분석 및 가장 신뢰도 높은 객체 선정
        max_conf = 0.0
        best_prediction = None

        # results[0]에 첫 번째 이미지에 대한 결과가 들어있습니다.
        result = results[0]
        for box in result.boxes:
            current_conf = float(box.conf[0])
            if current_conf > max_conf:
                max_conf = current_conf
                best_prediction = DetectionResult(
                    class_name=result.names[int(box.cls[0])],
                    confidence=round(current_conf, 4),
                    box=[int(coord) for coord in box.xyxy[0].tolist()]
                )

        if best_prediction is None:
            return PredictionResponse(status="no_item_detected", best_detection=None)
        
        return PredictionResponse(status="item_detected", best_detection=best_prediction)

    except Exception as e:
        # 추론 과정 중 발생한 모든 오류에 대해 500 에러 반환
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {str(e)}")

# ====================================================
# 5. 서버 실행 (로컬 개발용)
# ====================================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)