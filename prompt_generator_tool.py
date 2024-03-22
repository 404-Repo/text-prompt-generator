import PromptGenerator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="either set to 'online' or 'offline' or 'filter'.")
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
    else:
        raise ValueError(f"Unknown mode was specified: {args.mode}. Supported modes are 'online', 'offline', 'filter'.")

