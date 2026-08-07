"""Tests for the overlap-resolution logic used in Redactor.redact().

This tests the algorithm in isolation using lightweight fake entity
objects, so it runs without requiring presidio-analyzer/spaCy to be
installed -- useful for fast CI runs that don't need the full NLP
stack just to verify this specific logic is correct.
"""

from dataclasses import dataclass


@dataclass
class FakeEntity:
    entity_type: str
    start: int
    end: int
    score: float


def remove_overlaps(results):
    """Mirrors Redactor._remove_overlaps -- kept here as a standalone,
    directly testable copy of the algorithm. If you change the real
    implementation in veildoc/redaction.py, update this too (or
    refactor to share the function -- left separate here so this test
    file has zero import dependency on the presidio-analyzer package).
    """
    ranked = sorted(results, key=lambda r: (-r.score, -(r.end - r.start)))
    selected = []
    for r in ranked:
        overlaps = any(not (r.end <= s.start or r.start >= s.end) for s in selected)
        if not overlaps:
            selected.append(r)
    return selected


def test_non_overlapping_entities_all_kept():
    entities = [
        FakeEntity("PERSON", 0, 5, 0.9),
        FakeEntity("EMAIL_ADDRESS", 10, 20, 0.9),
    ]
    result = remove_overlaps(entities)
    assert len(result) == 2


def test_overlapping_entities_higher_confidence_wins():
    entities = [
        FakeEntity("ORGANIZATION", 0, 12, 0.7),   # lower confidence, overlaps below
        FakeEntity("GRANT_ID", 3, 18, 0.85),       # higher confidence
    ]
    result = remove_overlaps(entities)
    assert len(result) == 1
    assert result[0].entity_type == "GRANT_ID"


def test_overlapping_entities_equal_confidence_longer_wins():
    entities = [
        FakeEntity("ORGANIZATION", 0, 12, 0.85),   # length 12
        FakeEntity("GRANT_ID", 3, 25, 0.85),        # length 22, same confidence
    ]
    result = remove_overlaps(entities)
    assert len(result) == 1
    assert result[0].entity_type == "GRANT_ID"


def test_three_way_overlap_only_one_survives():
    entities = [
        FakeEntity("PERSON", 0, 10, 0.6),
        FakeEntity("ORGANIZATION", 2, 15, 0.7),
        FakeEntity("GRANT_ID", 5, 20, 0.9),
    ]
    result = remove_overlaps(entities)
    assert len(result) == 1
    assert result[0].entity_type == "GRANT_ID"


def test_adjacent_non_overlapping_entities_both_kept():
    # end of first == start of second: not overlapping (half-open intervals)
    entities = [
        FakeEntity("PERSON", 0, 10, 0.8),
        FakeEntity("EMAIL_ADDRESS", 10, 20, 0.8),
    ]
    result = remove_overlaps(entities)
    assert len(result) == 2
