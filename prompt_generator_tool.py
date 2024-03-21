import PromptGenerator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="either set to 'online' or 'offline'.")
    mode = parser.parse_args()

    prompt_generator = PromptGenerator.PromptGenerator()
    if mode == "online":
        prompt_generator.online_generator()
    elif mode == "offline":
        prompt_generator.offline_generator()
    else:
        raise ValueError(f"Unknown mode was specified: {mode}. Supported modes are 'online', 'offline'.")

    prompt_generator.filter_prompts()
