import argparse
from typing import Optional

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from openai import OpenAI
from openai.types.chat.chat_completion import ChoiceLogprobs


here = Path(__file__).parent

MIN_NUMBER = 0
MAX_NUMBER = 9

PROMPT = (
    f"Generate a random number between {MIN_NUMBER} and {MAX_NUMBER} (inclusive). "
    "Only output the number without additional text."
)


def generate_one_completion_and_get_logprobs(
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
) -> dict[int, float]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temperature,
        max_tokens=8,
        logprobs=True,
        top_logprobs=15,
        n=1,
    )
    logprobs: dict[int, float] = {i: -float("inf") for i in range(10)}
    choice = response.choices[0]
    assert isinstance(choice.logprobs, ChoiceLogprobs)
    assert choice.logprobs.content
    assert choice.logprobs.content[0].top_logprobs
    for item in choice.logprobs.content[0].top_logprobs:
        try:
            logprob = int(item.token)
            if logprob in logprobs:
                logprobs[logprob] = item.logprob
        except ValueError:
            raise ValueError("Invalid digit")
    return logprobs


def generate_n_completion_and_get_digits(
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    n: int = 100,
) -> list[int]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temperature,
        max_tokens=8,
        logprobs=True,
        top_logprobs=10,
        n=n,
    )
    digits: list[int] = []
    for choice in response.choices:
        try:
            assert choice.message.content
            digit = int(choice.message.content)
            digits.append(digit)
        except ValueError:
            raise ValueError("Invalid digit")
    return digits


def plots(
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    n: int = 100,
) -> None:
    _, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(MIN_NUMBER, MAX_NUMBER + 1)

    # Plot logprobs-based probabilities
    logprobs = generate_one_completion_and_get_logprobs(
        model, api_key, base_url, temperature
    )
    probs = np.exp(list(logprobs.values()))
    probs = probs / np.sum(probs)  # Normalize
    ax.bar(
        x=x - 0.2,
        height=probs,
        width=0.4,
        align="center",
        alpha=0.7,
        label="Logprobs",
    )

    # Plot empirical distributions
    digits = generate_n_completion_and_get_digits(
        model, api_key, base_url, temperature, n
    )
    values, counts = np.unique(digits, return_counts=True)
    emp_probs = counts / len(digits)
    full_probs = np.zeros(MAX_NUMBER - MIN_NUMBER + 1)
    for val, prob in zip(values, emp_probs):
        full_probs[val - 1] = prob
    ax.bar(
        x=x + 0.2,
        height=full_probs,
        width=0.4,
        align="center",
        alpha=0.7,
        label="Empirical",
    )

    ax.set_title(
        f"digits: [{MIN_NUMBER}, {MAX_NUMBER}] — temperature: {temperature:.1f} — n: {n} "
    )
    ax.set_xlabel("Digits")
    ax.set_ylabel("Probability")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_xticks(np.arange(MIN_NUMBER, MAX_NUMBER + 1))
    ax.legend(loc="upper left")
    plt.tight_layout()

    path = here / "figures" / f"{n:04d}" / f"{temperature:03.1f}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="Model identifier")
    parser.add_argument("--api-key", help="Provider API key", default="dummy-key")
    parser.add_argument("--base-url", help="API base URL for non-OpenAI endpoints")
    parser.add_argument("--n", type=int, default=100, help="Number of requests")
    args = parser.parse_args()

    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for temperature in temperatures:
        plots(
            args.model,
            args.api_key,
            args.base_url,
            temperature,
            n=args.n,
        )


if __name__ == "__main__":
    main()
