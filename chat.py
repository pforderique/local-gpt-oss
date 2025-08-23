"""Chat interface for the gpt-oss-20b model"""

import argparse
import dataclasses
import sys
import threading

from mlx_lm.utils import load
from openai_harmony import Message, Role

from llm.generate import stream_generate, DecodingControls
from llm.harmony import StarterConversations


USER_CLI_PROMPT = "User: "
ASSISTANT_CLI_PROMPT = "ChatGPT: "
QUIT_COMMANDS = ["exit", "quit", "q"]

FINAL_MESSAGE_SIGNAL = "final<|message|>"
THINKING_TEXT = "thinking"


class ThinkingAnimation:
    """A simple thinking animation that runs in a separate thread."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        dots = ["", ".", "..", "..."]
        i = 0
        # prefix already printed by caller
        while not self.stop_event.is_set():
            d = dots[i % len(dots)]
            pad = " " * (3 - len(d))  # overwrite previous longer state
            sys.stdout.write(f"\r{self.prefix}{THINKING_TEXT}{d}{pad}")
            sys.stdout.flush()
            i += 1
            self.stop_event.wait(0.3)

    def start(self):
        """Start the thinking animation."""
        self.thread.start()

    def stop(self):
        """Idempotent stop method for the thinking animation."""

        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.thread.join(timeout=1)
        # Clear the 'thinking...' area
        sys.stdout.write(f"\r{self.prefix}{' ' * (len(THINKING_TEXT) + 3)}\r{self.prefix}")
        sys.stdout.flush()


@dataclasses.dataclass
class ChatTurnMetrics:
    """Metrics for a single chat turn."""

    peak_memory: float
    generation_tps: float


def create_arg_parser():
    """Create the argument parser for the chat interface."""
    parser = argparse.ArgumentParser(description="ChatGPT OSS 20B")
    # display options
    parser.add_argument("--with_thoughts", action="store_true",
                        help="show thoughts during generation")
    parser.add_argument("--stream", action="store_true",
                        help="stream responses from the model")
    parser.add_argument("--no_animate", action="store_true",
                        help="disable animation of 'thinking...' while waiting")
    # decoding options
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="maximum number of tokens to generate")
    # defaults from https://lmstudio.ai/models/openai/gpt-oss-20b?utm_source=chatgpt.com
    parser.add_argument("--temp", type=float, default=0.7,
                        help="sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="nucleus sampling probability")
    parser.add_argument("--top_k", type=int, default=40,
                        help="top-k sampling")
    parser.add_argument("--min_p", type=float, default=0.05,
                        help="minimum probability")
    parser.add_argument("--repeat_penalty", type=float, default=1.1,
                        help="repeat penalty")
    return parser


def _print_header():
    """Print the header for the chat interface."""
    header_len = 15
    print("=" * header_len, "ChatGPT OSS 20B", "=" * header_len)
    print(f"Type '{QUIT_COMMANDS[0]}' to quit.")
    print()


def _write_response(response: str, animation: ThinkingAnimation) -> None:
    """Write the response to the standard output."""
    animation.stop()
    sys.stdout.write(response)
    sys.stdout.flush()


def main(args: argparse.Namespace) -> int:
    """Main function for the chat interface."""

    _print_header()
    print(f"Stream: {args.stream}")
    print(f"Show thoughts: {args.with_thoughts}")

    context_conversation = (
        StarterConversations.basic_system_with_developer_instructions(
            "Answer concisely."
        )
    )

    decoding_controls = DecodingControls(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        max_tokens=args.max_tokens,
    )

    print("Loading model and tokenizer...", end="", flush=True)
    model, tokenizer = load("models/gpt-oss-20b-4bit")
    print("loaded.")

    while (user_input := input(f"\n{USER_CLI_PROMPT}")) not in QUIT_COMMANDS:
        sys.stdout.write(ASSISTANT_CLI_PROMPT)
        sys.stdout.flush()

        animation = ThinkingAnimation(prefix=ASSISTANT_CLI_PROMPT)
        if not args.no_animate:
            animation.start()

        stream = stream_generate(
            model=model,
            tokenizer=tokenizer,
            user_input=user_input,
            context_conversation=context_conversation,
            decoding_controls=decoding_controls,
        )

        buf = []
        can_print_stream = args.with_thoughts
        for res in stream:
            text = res.text
            buf.append(text)
            _ = ChatTurnMetrics(
                peak_memory=res.peak_memory,
                generation_tps=res.generation_tps
            )

            if not args.stream:
                continue

            if can_print_stream:
                _write_response(text, animation)

            # Always stream print the final message
            if FINAL_MESSAGE_SIGNAL in "".join(buf[-2:]):
                can_print_stream = True

        full_response = "".join(buf)

        # TODO: use render convo to see if we are repeating assistant tags here
        # pylint: disable=no-member
        context_conversation.messages.append(
            Message.from_role_and_content(Role.ASSISTANT, full_response)
        )

        if args.stream:
            print()

        if args.with_thoughts:
            _write_response(full_response, animation)
            continue

        _, final_message = full_response.split(FINAL_MESSAGE_SIGNAL, maxsplit=1)
        _write_response(final_message, animation)

    return 0


if __name__ == "__main__":
    cli_args = create_arg_parser().parse_args()
    main(cli_args)
