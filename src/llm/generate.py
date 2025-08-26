"""Generate responses using the harmony"""

from typing import Iterator

from mlx import nn
from mlx_lm.generate import stream_generate as mlx_stream_generate, GenerationResponse
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.tokenizer_utils import TokenizerWrapper
from openai_harmony import Conversation
from pydantic import BaseModel, Field

from llm.harmony import harmonize_user_input


class DecodingControls(BaseModel):
    """Decoding controls for text generation."""

    # Defaults from https://lmstudio.ai/models/openai/gpt-oss-20b?utm_source=chatgpt.com
    temperature: float = Field(default=0.8, ge=0, le=1)
    top_p: float = Field(default=0.8, ge=0, le=1)
    top_k: int = Field(default=40, ge=0)
    min_p: float = Field(default=0.05, ge=0, le=1)
    repeat_penalty: float = Field(default=1.1, ge=1)

    max_tokens: int | None = Field(default=None, ge=1)


def stream_generate(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    user_input: str,
    context_conversation: Conversation | None = None,
    decoding_controls: DecodingControls | None = None,
    **kwargs,
) -> Iterator[GenerationResponse]:
    """
    Thin wrapper that:
      1) Formats user input using OpenAI Harmony
      2) Streams tokens from MLX-LM with a configurable sampler/logits processors

    Args:
      - model: The MLX-LM model to use for generation.
      - tokenizer: The tokenizer to use for encoding/decoding text.
      - user_input: The input text from the user.
      - context_conversation: An optional conversation context.
      - decoding_controls: Optional decoding controls.

    Yields:
      - str chunks as they are generated.
    """
    harmonized_prompt = harmonize_user_input(user_input, context_conversation)

    if decoding_controls is None:
        decoding_controls = DecodingControls()

    sampler = make_sampler(
        temp=decoding_controls.temperature,
        top_p=decoding_controls.top_p,
        top_k=decoding_controls.top_k,
        min_p=decoding_controls.min_p,
    )
    logits_processors = make_logits_processors(
        repetition_penalty=decoding_controls.repeat_penalty
    )

    for chunk in mlx_stream_generate(
        model,
        tokenizer,
        prompt=harmonized_prompt,
        max_tokens=decoding_controls.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        **kwargs
    ):
        if chunk:
            yield chunk


def generate(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    user_input: str,
    context_conversation: Conversation | None = None,
    decoding_controls: DecodingControls | None = None,
    **kwargs,
) -> str:
    """
    Thin wrapper that:
      1) Formats input via OpenAI Harmony
      2) Generates with MLX-LM using a configurable sampler/logits processors

    Args:
      - user_input: The input text from the user.
      - context_conversation: An optional conversation context.
      - decoding_controls: Optional decoding controls.

    Returns:
      - str: The generated response text.
    """
    return "".join(chunk.text for chunk in stream_generate(
        model,
        tokenizer,
        user_input,
        context_conversation=context_conversation,
        decoding_controls=decoding_controls,
        **kwargs
    ))
