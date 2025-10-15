"""Integrations with LLM clients and frameworks."""

from llmace.integrations.openai import inject_playbook_into_messages, ACEOpenAIWrapper

__all__ = ["inject_playbook_into_messages", "ACEOpenAIWrapper"]

