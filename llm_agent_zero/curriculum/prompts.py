curriculum_prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert competition-math problem setter.\n"
                "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                "Avoid re-using textbook clichés or famous contest problems.\n"
                "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                "<question>\n"
                "{The full problem statement on one or more lines}\n"
                "</question>\n\n"
                r"\boxed{final_answer}"
                "\n\n"
                "Do NOT output anything else—no explanations, no extra markup."
            )
        },
        {
            "role": "user",
            "content": (
                "Generate one new, challenging reasoning question now. "
                "Remember to format the output exactly as instructed."
            )
        }
    ]


executor_prompt = [
    {
        'role': 'system', 
        'content': (
    "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.\n"
    "First, provide your reasoning and write a self-contained Python code block wrapped in ```python ... ``` to help you calculate the answer. You must use the `print()` function to output the results.\n"
    "After you write the code block, STOP. I will execute it for you.\n"
    "I will then provide the output under 'Code execution result:'. You must use this result (even if it's an error) to continue your reasoning and provide the final answer.\n"
    "The final answer must be enclosed in \\boxed{...}."
    "Code Format:\n"
    "Each code snippet is wrapped between ```python and ```. You need to use print() to output intermediate results.\n"
    "Answer Format:\n"
    "The last part of your response should be: \\boxed{...}"
)
    }
    ]