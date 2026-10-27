from tests.conftest import FLAVOR_COLORS, USER_ID, make_pack, make_token


def _enroll(client, auth, slug, n=4, seed0=0):
    files = [("files", (f"{slug}{i}.jpg", make_pack(slug, seed=seed0 + i), "image/jpeg")) for i in range(n)]
    return client.post("/references", data={"flavor_id": f"id-{slug}"}, files=files, headers=auth)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_requires_auth(client):
    r = client.post("/predict", files={"file": ("a.jpg", make_pack("fresa"), "image/jpeg")})
    assert r.status_code == 401


def test_predict_rejects_bad_token(client):
    bad = {"Authorization": f"Bearer {make_token(secret='another-secret-that-is-long-enough!!')}"}
    r = client.post("/predict", files={"file": ("a.jpg", make_pack("fresa"), "image/jpeg")}, headers=bad)
    assert r.status_code == 401


def test_predict_rejects_expired_token(client):
    expired = {"Authorization": f"Bearer {make_token(exp_offset=-10)}"}
    r = client.post("/predict", files={"file": ("a.jpg", make_pack("fresa"), "image/jpeg")}, headers=expired)
    assert r.status_code == 401


def test_predict_without_references_returns_message(client, auth):
    r = client.post("/predict", files={"file": ("a.jpg", make_pack("fresa"), "image/jpeg")}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["predictions"] == []
    assert body["low_confidence"] is True
    assert "Sabores" in body["message"]


def test_enroll_then_predict_each_flavor(client, auth):
    for slug in FLAVOR_COLORS:
        r = _enroll(client, auth, slug)
        assert r.status_code == 201, r.text
        assert r.json()["total_for_flavor"] == 4

    correct = 0
    for slug in FLAVOR_COLORS:
        img = make_pack(slug, seed=100)
        r = client.post("/predict", files={"file": ("q.jpg", img, "image/jpeg")}, headers=auth)
        assert r.status_code == 200
        body = r.json()
        assert body["references_used"] == 24
        correct += body["predictions"][0]["slug"] == slug
    assert correct == len(FLAVOR_COLORS)


def test_stats_and_delete(client, auth, store):
    _enroll(client, auth, "coco", n=2)
    stats = client.get("/references/stats", headers=auth).json()
    coco = next(f for f in stats["flavors"] if f["slug"] == "coco")
    assert coco["references"] == 2

    rid = store.catalog(USER_ID).references[0].id
    assert client.delete(f"/references/{rid}", headers=auth).status_code == 204
    assert client.delete(f"/references/{rid}", headers=auth).status_code == 404


def test_enroll_unknown_flavor(client, auth):
    files = [("files", ("a.jpg", make_pack("fresa"), "image/jpeg"))]
    r = client.post("/references", data={"flavor_id": "nope"}, files=files, headers=auth)
    assert r.status_code == 404


def test_rejects_non_image(client, auth):
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")}, headers=auth)
    assert r.status_code == 415
    r = client.post("/predict", files={"file": ("a.jpg", b"hello", "image/jpeg")}, headers=auth)
    assert r.status_code == 422


def test_other_user_cannot_see_references(client, auth):
    _enroll(client, auth, "fresa", n=2)
    other = {"Authorization": f"Bearer {make_token(sub='22222222-2222-2222-2222-222222222222')}"}
    stats = client.get("/references/stats", headers=other).json()
    assert stats["flavors"] == []
