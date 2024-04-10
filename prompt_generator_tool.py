import tqdm

import PromptGenerator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="either set to 'online' or 'offline' or 'filter' or 'grammar'.")
    args = parser.parse_args()

    prompt_generator = PromptGenerator.PromptGenerator()
    if args.mode == "online":
        prompt_generator.online_generator()
        prompt_generator.filter_prompts()

    elif args.mode == "offline":
        prompt_generator.offline_generator()
        prompt_generator.filter_prompts()

    elif args.mode == "filter":
        prompt_generator.filter_prompts()

    elif args.mode == "grammar":
        prompts = prompt_generator.load_file_with_prompts()
        checked_prompts = prompt_generator.check_grammar(prompts)
        prompt_generator.save_prompts(checked_prompts, "w")

    elif args.mode == "semantic_check":
        prompts = prompt_generator.load_file_with_prompts()
        wrong_prompts = []
        correct_prompts = []
        for p, _ in zip(prompts, tqdm.trange(len(prompts))):
            score = prompt_generator.check_prompt(p)

            if float(score[0]) >= 0.5:
                correct_prompts.append(p)
            else:
                # print(f"prompt: {p}, score: {score}")
                wrong_prompts.append(p + ", [ " + score[0] + " ]")

        prompt_generator.save_prompts(correct_prompts, "w", file_name="correct_prompts.txt")
        prompt_generator.save_prompts(wrong_prompts, "w", file_name="wrong_prompts.txt")



    else:
        raise ValueError(f"Unknown mode was specified: {args.mode}. Supported modes are 'online', 'offline', 'filter', 'grammar.")

