from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


MODEL_PRICES_PER_1M = {
    "local-gold-baseline": (0.0, 0.0),
    "gemini-3.6-flash": (0.30, 2.50),
}


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3))


def estimate_cost(model: str, input_text: str, output_text: str) -> CostEstimate:
    input_price, output_price = MODEL_PRICES_PER_1M.get(model, (0.0, 0.0))
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return CostEstimate(input_tokens, output_tokens, cost)
