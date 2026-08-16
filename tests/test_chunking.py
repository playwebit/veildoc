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


def test_max_len_is_a_hard_cap_even_for_run_on_sentences():
    # regression test: a single "sentence" (no clean period+capital boundary,
    # e.g. due to abbreviations like "et al.") longer than max_len must still
    # be split, not passed through oversized. Found via real-world testing
    # on an academic PDF where citation-heavy prose exceeded max_len as one
    # sentence-boundary-detector "sentence".
    long_sentence = ("This method builds on prior work by Smith et al. and Jones et al., " * 15).strip()
    assert len(long_sentence) > 900

    chunks = chunk_text(long_sentence, min_len=10, max_len=900)
    assert len(chunks) > 1
    assert all(len(c) <= 900 for c in chunks), "max_len must be a hard cap"


def test_no_off_by_one_at_the_join_boundary():
    # regression test: found via real-world testing on an academic PDF where
    # a chunk came out at exactly max_len + 1 (901 instead of <=900). The bug:
    # the old check `len(buf) + len(sentence) <= max_len` didn't account for
    # the joining space added when concatenating buf + " " + sentence, so a
    # buf/sentence pair summing to exactly max_len produced a max_len+1 result.
    # This specific construction (two sentences summing to exactly 900) was
    # verified against the old buggy logic to reproduce a 901-length chunk.
    s1 = "Alpha " + ("x" * 439) + "."   # 446 chars
    s2 = "Beta " + ("y" * 448) + "."    # 454 chars -- 446 + 454 = 900 exactly
    text = s1 + " " + s2 + " Gamma next sentence follows here."

    chunks = chunk_text(text, min_len=10, max_len=900)
    assert all(len(c) <= 900 for c in chunks), (
        f"off-by-one regression: max chunk length {max(len(c) for c in chunks)} exceeds 900"
    )
