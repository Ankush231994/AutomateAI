import pytest

def build_prompt(system_prompt, rag_context, history, user_query):
    return f"{system_prompt}\n\nContext:\n{rag_context}\n\n{history}\nUser: {user_query}\nAssistant: "

def test_build_prompt():
    system_prompt = "You are Automate AI."
    rag_context = "Fact 1.\n\nFact 2."
    history = "User: Hi\nAssistant: Hello!"
    user_query = "What is Fact 1?"
    prompt = build_prompt(system_prompt, rag_context, history, user_query)
    expected = (
        "You are Automate AI.\n\nContext:\nFact 1.\n\nFact 2.\n\nUser: Hi\nAssistant: Hello!\nUser: What is Fact 1?\nAssistant: "
    )
    assert prompt == expected 