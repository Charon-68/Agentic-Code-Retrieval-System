"""Reusable prompt templates for Week 1 experiments."""

from __future__ import annotations

STRICT_JSON_SYSTEM_PROMPT: str = """You are a structured data extraction assistant.
You MUST respond with valid JSON only — no markdown fences, no commentary, no preamble.

Rules:
1. Output a single JSON object or array matching the user's schema exactly.
2. Use double quotes for all keys and string values.
3. If a field is unknown, use null rather than omitting required keys.
4. Never include trailing commas or comments in JSON.

Example valid response:
{"label": "positive", "confidence": 0.92, "reasoning": "The text expresses clear enthusiasm."}
"""

FEW_SHOT_CLASSIFICATION_TEMPLATE: str = """Classify the sentiment of the input text as one of: positive, negative, neutral.

Examples:
Text: "I absolutely love this product — best purchase ever!"
Label: positive

Text: "The delivery was late and the packaging was damaged."
Label: negative

Text: "The item arrived on Tuesday as scheduled."
Label: neutral

Text: "It's okay, nothing special but it works."
Label: neutral

Now classify the following text:
Text: "{input_text}"
Label:"""

CHAIN_OF_THOUGHT_MATH_TEMPLATE: str = """Solve the following math problem step by step.
Show your reasoning explicitly before giving the final numeric answer.

Problem: {problem}

Instructions:
1. Break the problem into clear logical steps.
2. Write each intermediate calculation on its own line.
3. End with a line that reads exactly: "Final Answer: <number>"

Begin your solution:
"""


def build_classification_prompt(input_text: str) -> str:
    """Format the few-shot classification template with user input."""
    return FEW_SHOT_CLASSIFICATION_TEMPLATE.format(input_text=input_text)


def build_cot_math_prompt(problem: str) -> str:
    """Format the chain-of-thought math reasoning template."""
    return CHAIN_OF_THOUGHT_MATH_TEMPLATE.format(problem=problem)
