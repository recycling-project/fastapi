from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import io
from PIL import Image
import uvicorn
import os
import pathlib
import torch  # GPU 사용 여부 확인용
# 서윤 배포하는 사이트에서 라이브러리 잘 깔도록 확인잘해 

# ====================================================
# 1. 환경 및 경로 설정 (수정된 경로 적용) 경로 재확인 필수 서윤
# ====================================================
BASE_DIR = pathlib.Path(__file__).parent.parent

MODEL_WEIGHTS_PATH = BASE_DIR / "recycling" / "ai" / "models" / "train_yolov8m" / "weights" / "best.pt"

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ----------------------------------------------------

# 2. FastAPI 애플리케이션 초기화
app = FastAPI(title="YOLOv8 Recycling Object Detection API")
global model


# 3. 서버 시작 시 모델 로드 (단 한 번, 메모리 효율성 확보)
@app.on_event("startup")
async def load_model():
    global model
    if not MODEL_WEIGHTS_PATH.exists():
        print(f"FATAL ERROR: Model file not found at {MODEL_WEIGHTS_PATH}")
        model = None
        return

    try:
        model = YOLO(MODEL_WEIGHTS_PATH)
        print("✅ YOLOv8 Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ Model Loading Error: {e}")
        model = None


# ----------------------------------------------------

# 4. 추론 API 엔드포인트 정의
@app.post("/predict/recycle_item")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialization failed.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    try:
        # 1. 이미지 파일 읽기 (비동기 처리)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. YOLO 모델로 예측 수행 (장치 및 Conf 명시)
        results = model.predict(
            source=image,
            conf=0.6,  # 확신도 60% 이상 객체만 탐지    # 서윤 확인 결과값확인하면서 수정
            iou=0.5,
            device=DEVICE,  # GPU/CPU 명시적으로 사용
            verbose=False
        )

        # 3. 결과 분석 및 핵심 아이템 선정 로직 (키오스크 서비스 최적화)
        max_conf = 0.0
        best_prediction = None

        for result in results:
            boxes = result.boxes
            names = result.names  # 클래스 이름 딕셔너리

            for box, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                current_conf = float(conf)

                # 가장 높은 확신도를 가진 객체만 최종 결과로 선정
                if current_conf > max_conf:
                    max_conf = current_conf
                    best_prediction = {
                        "class_name": names.get(int(cls), "Unknown"),  #서윤확인
                        "confidence": round(current_conf, 4),
                        # 바운딩 박스 좌표 (정수로 변환)
                        "box": [int(round(b)) for b in box]
                    }

        if best_prediction is None:
            return {"status": "no_item_detected", "detections": []}

        # 키오스크는 가장 확신 높은 하나의 아이템에 대한 안내가 필요하므로, 단일 객체만 반환
        return {"status": "item_detected", "best_detection": best_prediction}

    except Exception as e:
        # 추론 과정 중 발생한 오류 보고
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {str(e)}")


# ----------------------------------------------------

# 5. 서버 실행 (로컬 테스트용)
if __name__ == "__main__":
    # --reload 옵션은 코드를 수정할 때마다 서버를 자동 재시작합니다.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)