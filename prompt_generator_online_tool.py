import groq
import yaml
import tqdm
import re
import sys
import os
import colorama
import random
from time import time


if __name__ == '__main__':
    colorama.init()

    print("*" * 40)
    print("[INFO] *** Prompt Dataset Generator ***")
    print("*" * 40, "\n")

    with open("launching_config.yml", "r") as file:
        config_data = yaml.safe_load(file)

    # prompt for dataset generation
    prompt = config_data["prompt"]
    prompt = prompt.replace("prompts_num", str(config_data["prompts_num"]))

    print("[INFO] Input prompt: ")
    regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
    prompt_printing = regex.sub(r'\1\n', prompt)
    print(f"{colorama.Fore.GREEN}{prompt_printing}{colorama.Style.RESET_ALL}")

    # categories of the objects from where the objects will be piked up by the llm model
    object_categories = config_data['obj_categories']
    print(f"[INFO] Object categories: {colorama.Fore.GREEN}{object_categories}{colorama.Style.RESET_ALL}")

    print("[INFO] Started prompt generation.")
    t1 = time()

    client = groq.Groq(api_key=config_data["groq_api_key"])
    output_list = []

    for i in range(config_data["iteration_num"]):
        print(f"\n[INFO] Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL} \n")
        for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
            temperature = random.uniform(0.45, 0.65)
            print(f"[INFO] Current temperature: {colorama.Fore.GREEN}{temperature}{colorama.Style.RESET_ALL}")
            # find 'member' in the input string and replace it with category
            prompt = prompt.replace("member_placeholder", category)

            if config_data['llm_model']['seed'] < 0:
                seed = random.randint(0, sys.maxsize)
            else:
                seed = config_data['llm_model']['seed']

            output = client.chat.completions.create(messages=[
                                                {
                                                    "role": "system",
                                                    "content": "you are a helpful assistant. Be silent."
                                                },
                                                {
                                                    "role": "user",
                                                    "content": prompt + " Output format: python string with single prompt without numbers and ], [, (, ), {, }. "
                                                                        " Example: a red gorilla with green eyes. Single prompt per line. Avoid extra output."
                                                }
                                            ],
                                            model=config_data["groq_llm_model"],
                                            temperature=temperature,
                                            seed=seed,
                                            max_tokens=config_data["llm_model"]["max_tokens"])

            prompt = prompt.replace(category, "member_placeholder")

            # extracting the response of the llm model: generated prompts
            output_list.append(output.choices[0].message.content)

        t2 = time()
        duration = (t2 - t1) / 60.0
        print(f"[INFO] It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        print("[INFO] Done.")

        # pattern = r'[{}\[\]()\n"\'\\;/]'
        pattern = r'[^a-zA-Z\s]'
        result_prompts = []
        for el in output_list:
            lines = el.split(",")
            for i in range(len(lines)):
                lines[i] = re.sub(pattern, '', lines[i])
            result_prompts += lines

        # writing generated prompts to file
        with open(config_data["prompts_output_file"], "a") as file:
            for prompt in result_prompts:
                print(prompt)
                file.write("%s\n" % prompt)
