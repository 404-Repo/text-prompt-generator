import tqdm

import PromptGenerator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="either set to 'online' or 'offline' or "
                                                      "'filter' or 'grammar' or 'semantic_check_online' or 'semantic_check_offline.")
    args = parser.parse_args()

    prompt_generator = PromptGenerator.PromptGenerator()
    if args.mode == "online":
        prompt_generator.groq_generator()
        prompt_generator.filter_prompts()

    elif args.mode == "offline":
        prompt_generator.groq_generator()
        prompt_generator.filter_prompts()

    elif args.mode == "filter":
        prompt_generator.filter_prompts()

    elif args.mode == "grammar":
        prompts = prompt_generator.load_file_with_prompts()
        checked_prompts = prompt_generator.check_grammar(prompts)
        prompt_generator.save_prompts(checked_prompts, "w")

    elif args.mode == "semantic_check_offline":
        prompt_generator.transformers_load_checkpoint()
        prompts = prompt_generator.load_file_with_prompts()

        # edit [:] to give a range for check. It is fast, but not fast enough to check big dataset quicly
        for p, _ in zip(prompts[:], tqdm.trange(len(prompts[:]))):
            score = prompt_generator.transformers_check_prompt(p)

            if float(score) >= 0.5:
                p = prompt_generator.transformers_correct_prompt(p)
                p += "\n"
                prompt_generator.save_prompts([p], "a", file_name="correct_prompts.txt")
            else:
                p += ", [ " + score + " ]\n"
                prompt_generator.save_prompts([p], "a", file_name="wrong_prompts.txt")

    elif args.mode == "semantic_check_online":
        prompts = prompt_generator.load_file_with_prompts()
        # edit [:] to give a range for check. It is fast, but not fast enough to check big dataset quicly
        for p, _ in zip(prompts[:], tqdm.trange(len(prompts[:]))):
            score = prompt_generator.groq_check_prompt(p)

            if float(score) >= 0.5:
                p = prompt_generator.groq_correct_prompt(p)
                p += "\n"
                prompt_generator.save_prompts([p], "a", file_name="correct_prompts.txt")
            else:
                p += ", [ " + score + " ]\n"
                prompt_generator.save_prompts([p], "a", file_name="wrong_prompts.txt")

    else:
        raise ValueError(f"Unknown mode was specified: {args.mode}. Supported modes are 'online', 'offline', 'filter', 'grammar', 'semantic_check'.")

