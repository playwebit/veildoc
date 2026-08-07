from veildoc.chunking import chunk_text


def test_short_text_single_chunk():
    text = "This is a short paragraph that stays as one chunk."
    chunks = chunk_text(text, min_len=10, max_len=900)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_paragraph_split():
    text = "First paragraph here with enough length to pass the minimum.\n\nSecond paragraph also long enough to pass the minimum length check."
    chunks = chunk_text(text, min_len=10, max_len=900)
    assert len(chunks) == 2


def test_short_fragments_dropped():
    text = "Hi\n\nThis is a real paragraph that is long enough to be kept as a chunk on its own."
    chunks = chunk_text(text, min_len=20, max_len=900)
    assert len(chunks) == 1
    assert "Hi" not in chunks


def test_oversized_paragraph_split_at_sentences():
    sentence = "This is one sentence that repeats. "
    long_paragraph = sentence * 40  # will exceed max_len
    chunks = chunk_text(long_paragraph, min_len=10, max_len=200)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250  # allow small overflow from the last sentence added


def test_empty_text_returns_no_chunks():
    assert chunk_text("", min_len=10, max_len=900) == []


def test_chunks_preserve_order():
    text = "Alpha section text here, long enough.\n\nBeta section text here, long enough.\n\nGamma section text here, long enough."
    chunks = chunk_text(text, min_len=10, max_len=900)
    assert chunks[0].startswith("Alpha")
    assert chunks[1].startswith("Beta")
    assert chunks[2].startswith("Gamma")
