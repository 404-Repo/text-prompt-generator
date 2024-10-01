import re

from loguru import logger


def filter_unique_prompts(prompts: list[str]) -> list[str]:
    """
    Function for filtering all duplicates from the input prompt list

    Parameters
    ----------
    prompts: a list with input prompts

    Returns
    -------
    prompts: a list with unique prompts
    """
    logger.info("\n")
    logger.info("*" * 40)
    logger.info(" *** Filtering unique prompts. ***")
    logger.info("*" * 40)
    logger.info("\n")

    for i, p in enumerate(prompts):
        prompts[i] = " ".join(word.lower() for word in p.split())

    logger.info(f" Total lines in the dataset before: {len(prompts)}")

    articles = ["a", "the", "an"]
    prompts = [" ".join(word for word in sentence.split() if word.lower() not in articles) for sentence in prompts]
    prompts = list(set(prompts))
    prompts = [letter + "\n" if "\n" not in letter else letter for letter in prompts]

    logger.info(f" Total lines in the dataset after: {len(prompts)}")
    logger.info("\n")

    return prompts


def filter_prompts_with_words(prompts: list[str], words_to_filter: list[str]) -> list[str]:
    """
    Function that filters prompts with undesired words and prompts of certain length that might contain LLM bot output.

    Parameters
    ----------
    prompts: a list with input prompts
    words_to_filter: a list with words that will be used for filtering input prompts

    Returns
    -------
    prompts: list with filtered prompts
    """

    logger.info("\n")
    logger.info("*" * 40)
    logger.info(" *** Filtering prompts with undesired words. ***")
    logger.info("*" * 40)
    logger.info("\n")

    logger.info(f" Total lines in the dataset before: {len(prompts)}")

    prompts = list(filter(lambda sentence: 5 <= len(sentence) <= 100, prompts))
    prompts = list(
        filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(words_to_filter), prompts)
    )
    prompts = [letter + "\n" if "\n" not in letter else letter for letter in prompts]

    logger.info(f" Total lines in the dataset after: {len(prompts)}")
    logger.info("\n")

    return prompts


def correct_non_finished_prompts(prompts: list[str]) -> list[str]:
    """
    Function for correcting prompts that were not finished, e.g. generation in most cases stops on preposition.
    Removing that preposition still makes prompt valid.

    Parameters
    ----------
    prompts: a list with input prompts

    Returns
    -------
    filtered_prompts: list with filtered prompts
    """

    words = [
        "in",
        "on",
        "at",
        "by",
        "with",
        "about",
        "against",
        "among",
        "before",
        "behind",
        "between",
        "during",
        "for",
        "from",
        "of",
        "to",
        "over",
        "under",
        "through",
        "into",
        "upon",
        "within",
        "without",
        "along",
        "across",
        "behind",
        "beneath",
        "beside",
        "beyond",
        "near",
        "off",
        "onto",
        "towards",
        "underneath",
        "outside",
        "and",
        "that",
        "which",
    ]
    pattern = re.compile(r"\b(" + "|".join(words) + r")\b\s*$", re.IGNORECASE)
    filtered_prompts = [re.sub(pattern, "", prompt).strip() + "\n" for prompt in prompts]
    return filtered_prompts


def post_process_generated_prompts(prompts_list: list[str]) -> list[str]:
    """
    Function for post-processing of the generated prompts. The LLM output is filtered from punctuation symbols
    and all non-alphabetic characters.

    Parameters
    ----------
    prompts_list: a list with strings (generated prompts)

    Returns
    -------
    result_prompts: a list with processed prompts stored as strings.
    """

    result_prompts = []
    # for el in prompts_list:
    #     lines = el.split("\n")
    #     processed_lines = []
    #     for i in range(len(lines)):
    #         line = re.sub(r"[^a-zA-Z`\s-]", "", lines[i])
    #         line = re.sub(r"\d+", "", line)
    #         line = line.replace(".", "")
    #         line = line.replace("- ", "")

    #         if len(line.split()) > 3:
    #             if "\n" not in line:
    #                 line += "\n"
    #             processed_lines += [line]
    #     result_prompts += processed_lines
    for prompt in prompts_list:
        processed_lines = []
        for line in prompt.split("\n"):
            line = re.sub(r"[^a-zA-Z`\s-]", "", line)
            line = re.sub(r"\d+", "", line)
            line = line.strip()

            if len(line.split()) > 3:
                processed_lines.append(line + "\n")
        result_prompts.extend(processed_lines)
    return result_prompts


def remove_words_from_prompts(prompts: list[str], words_to_remove: list[str]) -> list[str]:
    """
    Function for removing words from the prompts

    Parameters
    ----------
    prompts: list of input prompts stored as strings
    words_to_remove: list of words that will be removed from the prompts of they will be found

    Returns
    -------
    result_prompts: a list with edited prompts
    """
    result_prompts = []
    for prompt in prompts:
        word_list = prompt.split()
        filtered_words = [word for word in word_list if word not in words_to_remove]
        result_prompt = " ".join(filtered_words)
        result_prompts.append(result_prompt)

    return result_prompts
