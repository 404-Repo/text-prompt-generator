import groq
import yaml
import tqdm
import re
import sys
import colorama
import random
import copy
from time import time


if __name__ == '__main__':
    colorama.init()

    print("*" * 40)
    print("[INFO] *** Prompt Dataset Generator ***")
    print("*" * 40, "\n")

    with open("launching_config.yml", "r") as file:
        config_data = yaml.safe_load(file)

    # prompt for dataset generation
    prompt = config_data["prompt_groq"]
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
        prompt_in = copy.copy(prompt)

        for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
            temperature = random.uniform(0.45, 0.65)
            print(f"[INFO] Current temperature: {colorama.Fore.GREEN}{temperature}{colorama.Style.RESET_ALL}")
            # find 'member' in the input string and replace it with category
            prompt_in = prompt_in.replace("member_placeholder", category)

            if config_data['llm_model']['seed'] < 0:
                seed = random.randint(0, sys.maxsize)
            else:
                seed = config_data['llm_model']['seed']

            output = client.chat.completions.create(messages=[
                                                {
                                                    "role": "user",
                                                    "content": prompt_in
                                                }
                                            ],
                                            model=config_data["groq_llm_model"],
                                            temperature=temperature,
                                            seed=seed,
                                            top_p=1,
                                            max_tokens=config_data["llm_model"]["max_tokens"])

            prompt_in = prompt_in.replace(category, "member_placeholder")

            # extracting the response of the llm model: generated prompts
            output_list.append(output.choices[0].message.content)

        pattern = r'[^a-zA-Z\s-]'
        result_prompts = []
        for el in output_list:
            lines = el.split(".")
            for i in range(len(lines)):
                lines[i] = re.sub(pattern, '', lines[i])
            result_prompts += lines

        # writing generated prompts to file
        with open(config_data["prompts_output_file"], "a") as file:
            for p in result_prompts:
                # print(prompt)
                file.write("%s" % p)

    t2 = time()
    duration = (t2 - t1) / 60.0
    print(f"[INFO] It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
    print("[INFO] Done.")
