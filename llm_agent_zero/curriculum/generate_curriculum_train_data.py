import pandas as pd
import argparse

def generate_curriculum_data(num_samples=100, output_path='./curriculum_train.parquet'):
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
    
    data_source = []
    prompt = []
    ability = []
    reward_model = []
    extra_info = []
    
    for idx in range(num_samples):
        data_source.append('math')
        prompt.append(curriculum_prompt)
        ability.append('math')
        reward_model.append({
            "style": "rule",
            "ground_truth": "",
        })
        
        extra_info.append({
            'split': 'train',
            'index': idx,
        })

    train_df = pd.DataFrame({
        'data_source': data_source,
        'prompt': prompt,
        'ability': ability,
        'reward_model': reward_model,
        'extra_info': extra_info
    })

    
    train_df.to_parquet(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a curriculum dataset.")
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=640, 
        help="Number of samples to generate (default: 100)"
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='data/curriculum_train.parquet', 
        help="Path to save the output parquet file (default: './curriculum_train.parquet')"
    )

    args = parser.parse_args()

    # Call the function with parsed arguments
    generate_curriculum_data(num_samples=args.num_samples, output_path=args.output_path)