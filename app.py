from flask import Flask, jsonify, Response, request
import json, joblib

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 연결 테스트용
@app.route('/api/test', methods=['GET'])
def test():
    data = {"message": "Flask 서버 연결 성공!"}
    return Response(json.dumps(data, ensure_ascii=False), content_type="application/json; charset=utf-8")

# 딥러닝 예측용
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text")

    # 🔹 임시 결과 반환 (모델 없어도 테스트용)
    result = "플라스틱" if "병" in text else "종이"
    return jsonify({"result": result})

if __name__ == '__main__':
    # model = joblib.load("models/recycling_model.pkl")  # 모델 없어도 실행 가능! 추후 모델 학습후 넣을예정
    app.run(host='0.0.0.0', port=5000)
