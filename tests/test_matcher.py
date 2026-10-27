import numpy as np

from app.ml.matcher import FlavorInfo, Reference, match

FLAVORS = [FlavorInfo("a", "fresa", "Fresa"), FlavorInfo("b", "pina", "Piña"), FlavorInfo("c", "coco", "Coco")]


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def hist(i, n=4):
    h = np.full(n, 0.01, np.float32)
    h[i] = 1
    return h / h.sum()


def refs():
    rng = np.random.default_rng(0)
    out = []
    for k, fid in enumerate(["a", "b", "c"]):
        base = np.eye(3)[k]
        for j in range(5):
            out.append(Reference(f"{fid}{j}", fid, unit(base + rng.normal(0, 0.05, 3)), hist(k)))
    return out


def test_match_picks_closest_flavor_with_high_confidence():
    res = match(unit([0.05, 1, 0.02]), hist(1), refs(), FLAVORS)
    assert res.predictions[0].slug == "pina"
    assert res.predictions[0].name == "Piña"
    assert not res.low_confidence
    assert res.references_used == 15
    assert len(res.predictions) == 3
    assert abs(sum(p.confidence for p in res.predictions) - 1) < 1e-3


def test_ambiguous_query_is_low_confidence():
    res = match(unit([1, 1, 0]), (hist(0) + hist(1)) / 2, refs(), FLAVORS)
    assert res.low_confidence


def test_no_references():
    res = match(unit([1, 0, 0]), hist(0), [], FLAVORS)
    assert res.predictions == []
    assert res.low_confidence
    assert res.references_used == 0


def test_references_of_inactive_flavors_are_ignored():
    res = match(unit([1, 0, 0]), hist(0), refs(), FLAVORS[1:])
    assert all(p.slug != "fresa" for p in res.predictions)
    assert res.references_used == 10
