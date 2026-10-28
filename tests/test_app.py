
import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/about")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}


def test_predict_valid_message(client):
    response = client.post(
        "/api/predict",
        json={"message": "Hey, are we meeting for lunch today?"},
    )

    assert response.status_code == 200

    data = response.get_json()
    assert data["prediction"] in ["Spam", "Ham"]
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 100


def test_predict_spam_message(client):
    response = client.post(
        "/api/predict",
        json={
            "message": "Congratulations! You have won a free prize. Call now!"
        },
    )

    assert response.status_code == 200

    data = response.get_json()
    assert data["prediction"] in ["Spam", "Ham"]
    assert 0 <= data["confidence"] <= 100


def test_predict_empty_message(client):
    response = client.post(
        "/api/predict",
        json={"message": "   "},
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_missing_message(client):
    response = client.post(
        "/api/predict",
        json={},
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_invalid_json(client):
    response = client.post(
        "/api/predict",
        data="not valid json",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "error" in response.get_json()
