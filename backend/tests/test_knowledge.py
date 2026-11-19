from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.modules.knowledge.models import Chunk
from app.worker import run_one


def test_search_before_anything_is_embedded(client: TestClient, db: Session) -> None:
    db.execute(delete(Chunk))  # undone after the test
    response = client.get("/api/v1/knowledge/search", params={"q": "bearing"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "KNOWLEDGE_NOTHING_EMBEDDED"


def test_add_text_embed_and_search(client: TestClient, db: Session) -> None:
    db.execute(delete(Chunk))  # undone after the test
    for title, text in [
        ("Rear axle bearing", "Remove the rear axle and replace the bearing. Torque 25 Nm."),
        ("Battery storage", "Store the battery at 50 percent charge in a dry room."),
    ]:
        response = client.post("/api/v1/knowledge/playground", json={"title": title, "text": text})
        assert response.status_code == 201

    while run_one(db):  # the worker embeds both texts (fake client, no API call)
        pass

    response = client.get("/api/v1/knowledge/search", params={"q": "rear axle bearing"})
    assert response.status_code == 200
    hits = response.json()["hits"]
    assert hits[0]["heading"] == "Playground > Rear axle bearing"
    assert hits[0]["text_rank"] == 1
