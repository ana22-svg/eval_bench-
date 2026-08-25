from eval_bench.judge import call_judge

# A case that should score high
r1 = call_judge("What is the capital of France?", "Paris", "Paris")
print("Expect ~5:", r1)

# A case that should score low
r2 = call_judge("What is the capital of France?", "Berlin", "Paris")
print("Expect ~1:", r2)
