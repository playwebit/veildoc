# veildoc

A client-side, minimal-disclosure privacy pipeline for using AI on sensitive
source documents (research papers, manuscripts, or any other document you
don't want fully exposed to a third-party AI provider).

Combines two mechanisms:

1. **Minimal-disclosure retrieval** — only the excerpt relevant to your
   question is ever selected for transmission, not the full document.
2. **Entity redaction** — names, emails, and other identifying information
   are stripped from that excerpt before it leaves your device, and restored
   locally in the AI's response.

Both stages run entirely on-device. Only the redacted, minimally-scoped
excerpt is ever transmitted — see the accompanying paper for the full
evaluation (20 real academic papers, 100 query-paper combinations) including
where this approach works well and where it doesn't (Section 5).

**This is research software accompanying an academic evaluation, not a
polished consumer product.** Read the Limitations section of the paper
before relying on it for anything sensitive.

## Install

```bash
pip install veildoc[local-embeddings]        # recommended: local embedding model
pip install veildoc[local-embeddings,anthropic]   # + Claude API support
pip install veildoc[all]                      # everything
```

You'll also need a spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Quick start — fully manual (no AI provider configured)

```python
from veildoc import Pipeline
from veildoc.embeddings import LocalBGEEmbedding

pipeline = Pipeline(embedding_backend=LocalBGEEmbedding())
pipeline.load_document("paper.pdf")

result = pipeline.ask("Can you help me improve the clarity of the methodology?")
print(f"Exposure: {result.exposure_pct:.1f}% of document")
print(result.redacted_excerpt)  # paste this into any chat AI yourself
```

## Quick start — end-to-end with an AI provider

```python
from veildoc import Pipeline
from veildoc.embeddings import LocalBGEEmbedding
from veildoc.llm import AnthropicBackend

pipeline = Pipeline(
    embedding_backend=LocalBGEEmbedding(),
    llm_backend=AnthropicBackend(api_key="sk-ant-..."),
)
pipeline.load_document("paper.pdf")
result = pipeline.ask("Can you help me improve the clarity of the methodology?")
print(result.restored_response)  # real names/values restored automatically
```

## Using a fully local/offline LLM (no network call at all)

```python
from veildoc.llm import OllamaBackend

pipeline = Pipeline(
    embedding_backend=LocalBGEEmbedding(),
    llm_backend=OllamaBackend(model="llama3"),  # requires `ollama run llama3` running locally
)
```

## Swapping in any other AI provider

Implement the three-method `LLMBackend` interface:

```python
from veildoc.llm.base import LLMBackend

class MyBackend(LLMBackend):
    def generate(self, prompt: str) -> str:
        # call whatever API or local model you want
        return my_model.generate(prompt)
```

Same pattern for embedding models via `EmbeddingBackend` — see
`veildoc/embeddings/base.py`.

## Command-line usage

```bash
veildoc ask paper.pdf "Improve the clarity of the methodology" --show-only
veildoc ask paper.pdf "Improve the clarity of the methodology" --llm anthropic --api-key sk-...
```

## What this does NOT protect against

- **Cumulative exposure across a multi-turn session.** Per-query exposure is
  low, but asking many questions about the same paper progressively exposes
  more of it. See the paper's Discussion section.
- **Stylometric deanonymization.** Redacting names doesn't hide writing
  style. See the paper's Section 5.6 for why our tested normalization
  approach didn't meaningfully help.
- **Retrieval accuracy is not guaranteed.** Our evaluation found the default
  local embedding model (BGE-base) fails to locate structurally distinct
  content (e.g. related-work sections) in a large majority of cases — see
  Section 5.3 of the paper before assuming retrieval always finds the right
  excerpt.

## Citation

If you use this in academic work, please cite the accompanying paper:

```
[Citation to be added once published]
```

## License

Apache-2.0
