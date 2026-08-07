"""Entity redaction: strip identifying information from text before
transmission, and restore it locally afterward using a token map that
never leaves the device.

Built on Presidio + spaCy, with three fixes validated during the
accompanying paper's evaluation on 20 real academic papers:

1. Overlap resolution -- Presidio can return overlapping entity spans
   (e.g. a generic ORGANIZATION match overlapping a specific regex
   match). Splicing these independently corrupts the output; we
   resolve conflicts by preferring higher-confidence, longer matches.
2. Technical-term allowlist + confidence floor -- fragmented PDF table
   text causes spaCy's NER to misread common abbreviations (e.g. "SGB",
   "HDL", "Table") as PERSON/ORGANIZATION entities. See the paper's
   Section 5.2 for the full discussion of this failure mode.
3. Broadened funding/grant-ID regex -- real papers use inconsistent
   funding-reference formats (e.g. "Grant No. 12345" vs.
   "DST/Reference.No.T-319/2018-19"); a single rigid pattern misses
   many of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_ALLOWLIST = {
    "sgb", "adb", "rf", "lr", "knn", "efnet", "cnn", "ml", "dl", "dnn", "rnn",
    "gpu", "cpu", "svm", "table", "perform", "reduce", "figure", "fig",
    "eq", "equation", "accuracy", "novel", "hdl", "vgg", "vgg16", "vgg19",
    "resnet", "googlenet", "alexnet", "densenet", "squeezenet", "inception",
    "mobilenet", "efficientnet", "lstm", "wpd", "cae", "kmeans", "k-means",
}

DEFAULT_TARGET_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "ORGANIZATION",
    "ORCID", "GRANT_ID", "DOI",
]

ORCID_REGEX = r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b"
GRANT_ID_REGEX = r"\b(?:Grant No\.?|Reference\.?\s?No\.?)\s?[A-Za-z0-9/\-]{4,40}\b"
DOI_REGEX = r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b"


@dataclass
class RedactionResult:
    redacted_text: str
    token_map: dict[str, str] = field(default_factory=dict)
    entity_count: int = 0


class Redactor:
    """Wraps Presidio + spaCy with the fixes described in the module
    docstring. Instantiate once and reuse across many calls to
    ``redact()`` -- constructing this loads the spaCy model, which is
    relatively slow.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        target_entities: list[str] | None = None,
        allowlist: set[str] | None = None,
        min_confidence: float = 0.5,
        extra_patterns: dict[str, str] | None = None,
    ):
        """
        Parameters
        ----------
        spacy_model:
            spaCy model name. The small model is faster and sufficient
            for this task; only switch to a larger model if you have a
            specific accuracy reason to.
        target_entities:
            Entity types to detect and redact. Defaults to
            ``DEFAULT_TARGET_ENTITIES``.
        allowlist:
            Lowercased terms that should never be redacted even if
            matched, used to suppress the false-positive pattern
            described in the module docstring. Defaults to
            ``DEFAULT_ALLOWLIST``; pass your own set to override, or
            an empty set to disable.
        min_confidence:
            Minimum Presidio confidence score to accept a match.
        extra_patterns:
            Additional regex patterns to register, as
            ``{entity_name: regex}``. Merged with the built-in
            ORCID/GRANT_ID/DOI patterns.
        """
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_analyzer.predefined_recognizers import SpacyRecognizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "presidio-analyzer is required. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            ) from exc

        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        ).create_engine()
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

        # ORGANIZATION isn't in Presidio's default registry -- must be
        # wired to spaCy's ORG label explicitly.
        self._analyzer.registry.add_recognizer(
            SpacyRecognizer(supported_entities=["ORGANIZATION"], supported_language="en")
        )

        patterns = {"ORCID": ORCID_REGEX, "GRANT_ID": GRANT_ID_REGEX, "DOI": DOI_REGEX}
        if extra_patterns:
            patterns.update(extra_patterns)
        for entity_name, regex in patterns.items():
            self._analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity=entity_name,
                    patterns=[Pattern(name=entity_name.lower(), regex=regex, score=0.85)],
                )
            )

        self.target_entities = target_entities or list(DEFAULT_TARGET_ENTITIES)
        self.allowlist = allowlist if allowlist is not None else set(DEFAULT_ALLOWLIST)
        self.min_confidence = min_confidence

    def _remove_overlaps(self, results):
        ranked = sorted(results, key=lambda r: (-r.score, -(r.end - r.start)))
        selected = []
        for r in ranked:
            overlaps = any(not (r.end <= s.start or r.start >= s.end) for s in selected)
            if not overlaps:
                selected.append(r)
        return selected

    def redact(self, text: str) -> RedactionResult:
        """Redact identifying entities from ``text``.

        Returns a ``RedactionResult`` with the redacted text and a
        token map (``{"[PERSON_1]": "Jane Doe", ...}``) for local
        restoration of an AI's response via ``restore()``. The token
        map should be kept in memory only and never transmitted.
        """
        raw_results = self._analyzer.analyze(text=text, language="en", entities=self.target_entities)
        filtered = [
            r for r in raw_results
            if text[r.start:r.end].strip().lower() not in self.allowlist
            and r.score >= self.min_confidence
        ]
        results = self._remove_overlaps(filtered)

        token_map: dict[str, str] = {}
        counters: dict[str, int] = {}
        redacted_text = text
        for r in sorted(results, key=lambda x: x.start, reverse=True):
            counters[r.entity_type] = counters.get(r.entity_type, 0) + 1
            token = f"[{r.entity_type}_{counters[r.entity_type]}]"
            token_map[token] = text[r.start:r.end]
            redacted_text = redacted_text[:r.start] + token + redacted_text[r.end:]

        return RedactionResult(
            redacted_text=redacted_text,
            token_map=token_map,
            entity_count=len(token_map),
        )


def restore(text: str, token_map: dict[str, str]) -> str:
    """Swap placeholder tokens in ``text`` back to their real values
    using ``token_map``. Use this on an AI's response after sending it
    redacted content, to display natural-language output to the user
    without the real values ever having left the device.
    """
    for token, original in token_map.items():
        text = text.replace(token, original)
    return text
