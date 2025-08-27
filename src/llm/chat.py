"""Chat interface for the gpt-oss-20b model"""

import argparse
import dataclasses
import sys
import threading
import time

from mlx_lm.utils import load
from openai_harmony import Message, Role

from llm.generate import stream_generate, DecodingControls
from llm.harmony import Conversation, StarterConversations


USER_CLI_PROMPT = "User: "
ASSISTANT_CLI_PROMPT = "ChatGPT: "
QUIT_COMMANDS = ["exit", "quit", "q"]

FINAL_MESSAGE_SIGNAL = "final<|message|>"
THINKING_TEXT = "thinking"


@dataclasses.dataclass
class ChatTurnMetrics:
    """Metrics for a single chat turn."""

    latency: float
    peak_memory: float
    generation_tps: float
    total_tokens: int
    first_token_latency: float


@dataclasses.dataclass
class DisplayOptions:
    """Options for displaying the chat output."""

    show_thoughts: bool = False
    show_metrics: bool = False
    no_stream: bool = False
    no_animate: bool = False


class _ThinkingAnimation:
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

        if self.thread.is_alive():
            self.thread.join(timeout=1)
        # Clear the 'thinking...' area
        sys.stdout.write(
            f"\r{self.prefix}{' ' * (len(THINKING_TEXT) + 3)}\r{self.prefix}")
        sys.stdout.flush()


class Chat:
    """A simple chat interface for interacting with the model."""

    def __init__(self,
                 model_path: str,
                 context_conversation: Conversation,
                 display_options: DisplayOptions,
                 decoding_controls: DecodingControls | None = None,
                 ):
        self.model, self.tokenizer = load(model_path)
        self.context_conversation = context_conversation
        self.display_options = display_options
        self.decoding_controls = decoding_controls

        self.animation = _ThinkingAnimation(prefix=ASSISTANT_CLI_PROMPT)

    def run(self) -> None:
        """Run the chat interface."""
        while (user_input := input(f"\n{USER_CLI_PROMPT}")) not in QUIT_COMMANDS:
            sys.stdout.write(ASSISTANT_CLI_PROMPT)
            sys.stdout.flush()

            self.animation = _ThinkingAnimation(prefix=ASSISTANT_CLI_PROMPT)
            if not self.display_options.no_animate:
                self.animation.start()

            stream = stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                user_input=user_input,
                context_conversation=self.context_conversation,
                decoding_controls=self.decoding_controls,
            )

            buf = []
            can_print_stream = self.display_options.show_thoughts
            metrics = ChatTurnMetrics(0, 0, 0, 0, 0)
            start_time = time.perf_counter()
            for res in stream:
                text = res.text
                buf.append(text)

                if len(buf) == 1:
                    metrics.first_token_latency = time.perf_counter() - start_time

                if self.display_options.no_stream:
                    continue

                if can_print_stream:
                    self._write_response(text)

                # Always stream print the final message
                if FINAL_MESSAGE_SIGNAL in "".join(buf[-2:]):
                    can_print_stream = True

            end_time = time.perf_counter()
            metrics.latency = end_time - start_time
            metrics.total_tokens = len(buf)
            metrics.generation_tps = (len(buf) / (end_time - start_time))
            # pylint: disable=undefined-loop-variable
            metrics.peak_memory = res.peak_memory if res else 0

            full_response = "".join(buf)
            print(buf)

            # pylint: disable=no-member
            self.context_conversation.messages.append(
                Message.from_role_and_content(Role.ASSISTANT, full_response)
            )

            metrics_data = ""
            if self.display_options.show_metrics:
                metrics_data = (
                    f"\n({metrics.total_tokens} tokens in {metrics.latency:.2f}s"
                    f" | peak_memory={metrics.peak_memory:.2f}Gb"
                    f" | tokens/sec={metrics.generation_tps:.2f}"
                    f" | first_token_latency={metrics.first_token_latency:.2f}s)"
                )

            if not self.display_options.no_stream:
                print(metrics_data)
                continue

            if self.display_options.show_thoughts:
                self._write_response(full_response)
                print(metrics_data)
                continue

            split = full_response.split(FINAL_MESSAGE_SIGNAL, maxsplit=1)
            if len(split) != 2:
                self._write_response("<no response>")
            else:
                final_message = split[1]
                self._write_response(final_message)
            print(metrics_data)

    def _write_response(self, response: str) -> None:
        """Write the response to the standard output."""
        self.animation.stop()
        sys.stdout.write(response)
        sys.stdout.flush()


def _create_arg_parser():
    """Create the argument parser for the chat interface."""
    parser = argparse.ArgumentParser(description="ChatGPT OSS 20B")
    # display options
    parser.add_argument("--show_thoughts", action="store_true",
                        help="show thoughts during generation")
    parser.add_argument("--show_metrics", action="store_true",
                        help="show generation metrics")
    parser.add_argument("--no_stream", action="store_true",
                        help="disable streaming responses from the model")
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


def main(args: argparse.Namespace) -> None:
    """Main function for the chat interface."""

    _print_header()
    print("Settings:")
    for option, value in vars(args).items():
        print(f"> {option}: {value}")

    context_conversation = (
        StarterConversations.basic_system_with_developer_instructions(
            "Answer concisely."
        )
    )

    display_options = DisplayOptions(
        show_thoughts=args.show_thoughts,
        show_metrics=args.show_metrics,
        no_stream=args.no_stream,
        no_animate=args.no_animate,
    )

    decoding_controls = DecodingControls(
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repeat_penalty=args.repeat_penalty,
        max_tokens=args.max_tokens,
    )

    print("Loading model...", end="", flush=True)
    chat = Chat(
        model_path="models/gpt-oss-20b-4bit",
        context_conversation=context_conversation,
        display_options=display_options,
        decoding_controls=decoding_controls,
    )
    print("done.")

    return chat.run()


if __name__ == "__main__":
    cli_args = _create_arg_parser().parse_args()
    main(cli_args)
