from veildoc.redaction import restore


def test_restore_single_token():
    text = "Contact [EMAIL_ADDRESS_1] for details."
    token_map = {"[EMAIL_ADDRESS_1]": "jane@example.com"}
    assert restore(text, token_map) == "Contact jane@example.com for details."


def test_restore_multiple_tokens():
    text = "[PERSON_1] worked with [PERSON_2] at [ORGANIZATION_1]."
    token_map = {
        "[PERSON_1]": "Alice",
        "[PERSON_2]": "Bob",
        "[ORGANIZATION_1]": "Acme University",
    }
    assert restore(text, token_map) == "Alice worked with Bob at Acme University."


def test_restore_no_tokens_present():
    text = "This text has no placeholders at all."
    assert restore(text, {"[PERSON_1]": "Alice"}) == text


def test_restore_empty_token_map():
    text = "[PERSON_1] said hello."
    assert restore(text, {}) == text


def test_restore_repeated_token():
    text = "[PERSON_1] met [PERSON_1] again."
    token_map = {"[PERSON_1]": "Alice"}
    assert restore(text, token_map) == "Alice met Alice again."
