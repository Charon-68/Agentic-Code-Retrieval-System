"""Week 1 CLI experiments: temperature sampling and Lost-in-the-Middle demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, Optional

import tiktoken
from openai import OpenAI

from src.week1_foundations.config import get_openai_client, get_settings, is_openai_configured
from src.week1_foundations.prompts import (
    STRICT_JSON_SYSTEM_PROMPT,
    build_classification_prompt,
    build_cot_math_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_SENTENCE = (
    "The secret retrieval code for this experiment is: NEBULA-7742."
)
FILLER_PARAGRAPH = (
    "This is filler context used to simulate a long document window. "
    "It contains no actionable information and exists only to push the "
    "target sentence away from the beginning and end of the context. "
    "Researchers have observed that language models often struggle to "
    "recall facts placed in the middle of very long prompts. "
)


def _get_encoding(model: str = "gpt-4o-mini") -> tiktoken.Encoding:
    """Return the tiktoken encoding for the given model."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def build_lost_in_the_middle_context(
    target: str = TARGET_SENTENCE,
    total_tokens: int = 4000,
    model: str = "gpt-4o-mini",
) -> tuple[str, int]:
    """
    Build a long dummy context with the target sentence inserted near the middle.

    Returns the full context string and the token index where the target begins.
    """
    encoding = _get_encoding(model)
    filler_tokens = encoding.encode(FILLER_PARAGRAPH)
    target_tokens = encoding.encode(target)

    if len(target_tokens) >= total_tokens:
        raise ValueError("Target sentence alone exceeds total_tokens budget.")

    remaining = total_tokens - len(target_tokens)
    half = remaining // 2

    prefix_ids = (filler_tokens * ((half // len(filler_tokens)) + 1))[:half]
    suffix_ids = (filler_tokens * ((remaining - half) // len(filler_tokens) + 1))[
        : remaining - half
    ]

    full_ids = prefix_ids + target_tokens + suffix_ids
    full_ids = full_ids[:total_tokens]
    context = encoding.decode(full_ids)
    target_position = len(prefix_ids)
    return context, target_position


def run_temperature_comparison(
    client: OpenAI,
    prompt: str,
    model: str,
    temperatures: tuple[float, ...] = (0.0, 1.0),
    n_samples: int = 3,
) -> dict[str, Any]:
    """
    Run the same prompt at different temperatures and log response variation.

    At temperature 0.0 responses should be nearly deterministic; at 1.0 they
    should show greater lexical diversity across samples.
    """
    results: dict[str, Any] = {}

    for temp in temperatures:
        samples: list[dict[str, Any]] = []
        for i in range(n_samples):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=150,
                logprobs=True,
                top_logprobs=3,
            )
            choice = response.choices[0]
            top_logprobs_summary: list[dict[str, Any]] = []
            if choice.logprobs and choice.logprobs.content:
                for token_info in choice.logprobs.content[:5]:
                    alts = [
                        {"token": alt.token, "logprob": alt.logprob}
                        for alt in (token_info.top_logprobs or [])
                    ]
                    top_logprobs_summary.append(
                        {
                            "token": token_info.token,
                            "logprob": token_info.logprob,
                            "alternatives": alts,
                        }
                    )

            samples.append(
                {
                    "sample_index": i,
                    "text": choice.message.content or "",
                    "finish_reason": choice.finish_reason,
                    "top_token_logprobs": top_logprobs_summary,
                }
            )
            logger.info(
                "temperature=%.1f sample=%d | %s",
                temp,
                i,
                (choice.message.content or "")[:120],
            )

        unique_responses = len({s["text"] for s in samples})
        results[str(temp)] = {
            "samples": samples,
            "unique_response_count": unique_responses,
            "interpretation": (
                "Deterministic (low diversity)"
                if temp == 0.0 and unique_responses <= 1
                else "Higher variance expected at elevated temperature"
            ),
        }

    return results


def demonstrate_lost_in_the_middle(
    client: OpenAI,
    model: str,
    question: str = "What is the secret retrieval code mentioned in the document?",
) -> dict[str, Any]:
    """
    Query an LLM with a long context where the answer sits in the middle.

    Models frequently fail to recall mid-context facts — this reproduces that effect.
    """
    context, target_token_index = build_lost_in_the_middle_context(model=model)
    encoding = _get_encoding(model)
    total_tokens = len(encoding.encode(context))

    logger.info(
        "Built context: %d tokens, target inserted at token index ~%d",
        total_tokens,
        target_token_index,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions using ONLY the provided document. "
                    "If the answer is not present, say 'NOT FOUND'."
                ),
            },
            {
                "role": "user",
                "content": f"Document:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.0,
        max_tokens=100,
    )

    answer = response.choices[0].message.content or ""
    found = "NEBULA-7742" in answer

    result = {
        "total_context_tokens": total_tokens,
        "target_token_index": target_token_index,
        "question": question,
        "model_answer": answer,
        "target_recalled": found,
        "expected_code": "NEBULA-7742",
    }
    logger.info("Lost-in-the-Middle result | recalled=%s | answer=%s", found, answer)
    return result


async def run_async_classification_demo(
    client: OpenAI,
    model: str,
    text: str,
) -> str:
    """Async wrapper demonstrating structured JSON output with few-shot prompting."""
    loop = asyncio.get_event_loop()
    prompt = build_classification_prompt(text)

    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STRICT_JSON_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        'Respond as JSON: {{"label": "<positive|negative|neutral>", '
                        '"confidence": <float>}}'
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    return await loop.run_in_executor(None, _call)


def run_cot_math_demo(client: OpenAI, model: str) -> str:
    """Run a chain-of-thought math reasoning example."""
    problem = (
        "A bookstore sells novels for $12 each and comics for $8 each. "
        "If Maya buys 3 novels and 5 comics, how much does she spend in total?"
    )
    prompt = build_cot_math_prompt(problem)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )
    return response.choices[0].message.content or ""


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for Week 1 experiments."""
    parser = argparse.ArgumentParser(description="Week 1 LLM Foundations Experiments")
    parser.add_argument(
        "--task",
        choices=["temperature", "lost-in-middle", "classification", "cot-math", "all"],
        default="all",
        help="Which experiment to run",
    )
    args = parser.parse_args(argv)

    if not is_openai_configured():
        logger.error("OPENAI_API_KEY not configured. See .env.example")
        return 1

    settings = get_settings()
    client = get_openai_client()
    model = settings.openai_model

    if args.task in ("temperature", "all"):
        print("\n=== Temperature Comparison (0.0 vs 1.0) ===\n")
        prompt = "Write a one-sentence tagline for a coffee shop called 'Midnight Roast'."
        temp_results = run_temperature_comparison(client, prompt, model)
        print(json.dumps(temp_results, indent=2))

    if args.task in ("lost-in-middle", "all"):
        print("\n=== Lost in the Middle Demo ===\n")
        litm = demonstrate_lost_in_the_middle(client, model)
        print(json.dumps(litm, indent=2))

    if args.task in ("classification", "all"):
        print("\n=== Few-Shot Classification (JSON) ===\n")
        result = asyncio.run(
            run_async_classification_demo(
                client,
                model,
                "The support team resolved my issue within minutes — fantastic service!",
            )
        )
        print(result)

    if args.task in ("cot-math", "all"):
        print("\n=== Chain-of-Thought Math ===\n")
        print(run_cot_math_demo(client, model))

    return 0


if __name__ == "__main__":
    sys.exit(main())
