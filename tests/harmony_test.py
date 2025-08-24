import llm.harmony as harmony


_SYSTEM_ROLE_SIGNAL = "system"
_USER_ROLE_SIGNAL = "user"
_ASSISTANT_ROLE_SIGNAL = "assistant"
_ALL_SIGNALS = [
    _SYSTEM_ROLE_SIGNAL,
    _USER_ROLE_SIGNAL,
    _ASSISTANT_ROLE_SIGNAL
]

def test_harmonize_user_input_with_no_conversation():
    user_text = "Hello there"

    out = harmony.harmonize_user_input(user_text, conversation=None)

    for signal in _ALL_SIGNALS:
        assert signal in out


def test_harmonize_user_input_with_existing_conversation():
    user_text = "Hello again"

    conv = harmony.StarterConversations.basic_system_only()
    out = harmony.harmonize_user_input(user_text, conversation=conv)

    for signal in _ALL_SIGNALS:
        assert signal in out

    print(conv)
    for role in (harmony.Role.SYSTEM, harmony.Role.USER):
        assert any(msg.author.role == role for msg in conv.messages)
