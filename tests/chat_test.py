import builtins
from types import SimpleNamespace
from unittest.mock import create_autospec

from mlx import nn
from mlx.core import array
from mlx_lm.generate import GenerationResponse
from mlx_lm.tokenizer_utils import TokenizerWrapper
import pytest
from openai_harmony import Conversation

import llm.chat as chat


def _resp(text: str) -> GenerationResponse:
    return GenerationResponse(
        text=text,
        token=0,
        logprobs=array([0]),
        from_draft=False,
        prompt_tokens=0,
        peak_memory=0.1,
        generation_tps=5.0,
        prompt_tps=1.0,
        generation_tokens=1,
    )


@pytest.fixture
def fake_conv():
    return Conversation(messages=[])


@pytest.fixture
def patch_message_factory(monkeypatch):
    def _factory(role, content):
        return SimpleNamespace(role=role, content=content)

    monkeypatch.setattr(
        chat, "Message",
        SimpleNamespace(from_role_and_content=lambda role, content: _factory(role, content))
    )

@pytest.fixture
def mock_model():
    return create_autospec(nn.Module, instance=True)


@pytest.fixture
def mock_tokenizer():
    return create_autospec(TokenizerWrapper, instance=True)


@pytest.fixture
def mock_load(monkeypatch, mock_model, mock_tokenizer):
    monkeypatch.setattr(chat, "load", lambda path: (mock_model, mock_tokenizer))
    return mock_model, mock_tokenizer


def test_chat_run_only_prints_final_message(
    monkeypatch, capsys, fake_conv, patch_message_factory, mock_load
):
    chunks = [
        _resp("thoughts"),
        _resp(chat.FINAL_MESSAGE_SIGNAL),
        _resp("answer"),
    ]
    monkeypatch.setattr(chat, "stream_generate", lambda **_: iter(chunks))

    display = chat.DisplayOptions(
        show_thoughts=False, show_metrics=False, no_stream=True, no_animate=True
    )
    decoding = chat.DecodingControls()

    inputs = iter(["hello", "quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    chat.Chat(
        model_path="models/gpt-oss-20b-4bit",
        context_conversation=fake_conv,
        display_options=display,
        decoding_controls=decoding,
    ).run()

    out = capsys.readouterr().out
    assert chat.FINAL_MESSAGE_SIGNAL not in out
    assert "ChatGPT: answer" in out


def test_chat_run_prints_thoughts(
    monkeypatch, capsys, fake_conv, patch_message_factory, mock_load
):
    chunks = [
        _resp("thoughts"),
        _resp(chat.FINAL_MESSAGE_SIGNAL),
        _resp("answer"),
    ]
    monkeypatch.setattr(chat, "stream_generate", lambda **_: iter(chunks))

    display = chat.DisplayOptions(
        show_thoughts=True, show_metrics=False, no_stream=True, no_animate=True
    )

    inputs = iter(["hi", "quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    chat.Chat(
        model_path="models/gpt-oss-20b-4bit",
        context_conversation=fake_conv,
        display_options=display,
        decoding_controls=None,
    ).run()
    out = capsys.readouterr().out

    assert "thoughts" in out
    assert chat.FINAL_MESSAGE_SIGNAL in out
    assert "answer" in out


def test_chat_run_prints_metrics(
    monkeypatch, capsys, fake_conv, patch_message_factory, mock_load
):
    chunks = [
        _resp("thoughts"),
        _resp(chat.FINAL_MESSAGE_SIGNAL),
        _resp("answer"),
    ]
    monkeypatch.setattr(chat, "stream_generate", lambda **_: iter(chunks))

    display = chat.DisplayOptions(
        show_thoughts=False, show_metrics=True, no_stream=True, no_animate=True
    )

    inputs = iter(["hi", "quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    chat.Chat(
        model_path="models/gpt-oss-20b-4bit",
        context_conversation=fake_conv,
        display_options=display,
        decoding_controls=None,
    ).run()
    out = capsys.readouterr().out

    assert "thoughts" not in out
    assert chat.FINAL_MESSAGE_SIGNAL not in out
    assert "answer" in out
