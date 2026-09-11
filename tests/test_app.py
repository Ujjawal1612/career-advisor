import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app import app


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_prediction(monkeypatch, tmp_path):
    features = ["sslc", "hsc", "cgpa"]
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    y = np.array([0, 0, 1, 1, 1])
    model.fit(X, y)

    class Encoder:
        classes_ = np.array(["Data Analyst", "Software Developer"])

    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "label_encoder": Encoder(), "features": features, "accuracy": 1.0}, path)
    monkeypatch.setattr("app.MODEL_PATH", path)

    client = app.test_client()
    response = client.post("/predict", json={"sslc": 5, "hsc": 5, "cgpa": 5})
    assert response.status_code == 200
    assert "recommendation" in response.get_json()
    assert len(response.get_json()["ranking"]) == 2
