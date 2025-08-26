from unittest.mock import create_autospec

from mlx import nn
from mlx.core import array
from mlx_lm.generate import GenerationResponse
from mlx_lm.tokenizer_utils import TokenizerWrapper
import pytest

import llm.generate as gen


_FAKE_STREAM_TOKENS = ["hello", "how", "are", "you"]

@pytest.fixture
def mock_model():
    return create_autospec(nn.Module, instance=True)


@pytest.fixture
def mock_tokenizer():
    return create_autospec(TokenizerWrapper, instance=True)


def fake_generation(text: str):
    return GenerationResponse(
        text=text,
        token=0,
        logprobs=array([0]),
        from_draft=False,
        prompt_tokens=0,
        peak_memory=0.1,
        generation_tps=5.0,
        prompt_tps=1,
        generation_tokens=0,
    )


def fake_mlx_stream_generate(model, tokenizer, prompt, **kwargs):
    del model
    del tokenizer
    del prompt
    chunks = [fake_generation(t) for t in _FAKE_STREAM_TOKENS]
    for chunk in chunks:
        yield chunk


def test_stream_generate_yields_chunks(monkeypatch, mock_model, mock_tokenizer):
    monkeypatch.setattr(gen, "mlx_stream_generate", fake_mlx_stream_generate)

    chunks = list(gen.stream_generate(mock_model, mock_tokenizer, user_input="hi"))
    assert [c.text for c in chunks] == _FAKE_STREAM_TOKENS

def test_generate_concatenates(monkeypatch, mock_model, mock_tokenizer):
    monkeypatch.setattr(gen, "mlx_stream_generate", fake_mlx_stream_generate)

    out = gen.generate(mock_model, mock_tokenizer, user_input="hi")
    assert out == "hellohowareyou"

def test_decoding_controls_passed(monkeypatch, mock_model, mock_tokenizer):
    captured = {}
    def capture_stream(model, tokenizer, prompt, **kwargs):
        captured.update(kwargs)
        yield fake_generation("ok")

    monkeypatch.setattr(gen, "mlx_stream_generate", capture_stream)

    dc = gen.DecodingControls(max_tokens=7, temperature=0.5)
    gen.generate(mock_model, mock_tokenizer, "hi", decoding_controls=dc)

    assert captured["max_tokens"] == 7
    assert "sampler" in captured and "logits_processors" in captured
