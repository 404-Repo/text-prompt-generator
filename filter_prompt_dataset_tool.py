import yaml
import colorama


if __name__ == '__main__':
    colorama.init()

    print("*" * 40)
    print("[INFO] *** Prompt Dataset Cleaner ***")
    print("*" * 40, "\n")

    with open("launching_config.yml", "r") as file:
        config_data = yaml.safe_load(file)

    with open(config_data["prompts_output_file"], "r") as file:
        lines = [line.rstrip() for line in file]
        for i, l in enumerate(lines):
            lines[i] = ' '.join(word.lower() for word in l.split())


    print(f"[INFO] Total lines in the dataset before: {colorama.Fore.GREEN}{ len(lines)}{colorama.Style.RESET_ALL}")
    articles = ["a", "the", "an"]
    lines = [' '.join(word for word in sentence.split() if word.lower() not in articles) for sentence in lines]
    lines = list(set(lines))
    lines = list(filter(lambda sentence: 3 <= len(sentence) <= float("inf"), lines))
    lines = list(filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(config_data["filter_prompts_with_words"]), lines))
    print(f"[INFO] Total lines in the dataset after: {colorama.Fore.GREEN}{ len(lines)}{colorama.Style.RESET_ALL}")

    with open(config_data["prompts_output_file"], "w") as file:
        for prompt in lines:
            file.write("%s\n" % prompt)
