"""Thin wrapper around OpenAI Harmony API for easier usage"""

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
)


_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def harmonize_user_input(
    user_input: str,
    conversation: Conversation | None = None
) -> str:
    """
    Harmonize user input for the OpenAI Harmony API.

    Appends the user input to the conversation context and returns the full
    conversation context as a string.

    Args:
        user_input: The user input to harmonize.
        conversation: The current conversation context.

    Returns:
        str: The harmonized user input as a large prompt.
    """
    if not conversation:
        conversation = StarterConversations.basic_system_only()

    conversation.messages.append(
        Message.from_role_and_content(Role.USER, user_input)
    )

    return _enc.decode_utf8(
        _enc.render_conversation_for_completion(conversation, Role.ASSISTANT)
    )


class StarterConversations:
    """Starter conversations for the OpenAI Harmony API."""

    @staticmethod
    def basic_system_only() -> Conversation:
        """
        Provide the default system message for the OpenAI Harmony API.
        """
        return Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        ])

    @staticmethod
    def basic_system_with_developer_instructions(instructions: str) -> Conversation:
        """
        Provide the default system message with developer instructions for the OpenAI Harmony API.
        """
        return Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(instructions)
            )
        ])
