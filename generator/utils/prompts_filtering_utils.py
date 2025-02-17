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
    prompts_before = len(prompts)
    prompts = list(set(prompts))
    logger.info(f"Filtered repeated words. Prompts before: {prompts_before}. Prompts after: {len(prompts)}")

    return prompts


def filter_prompts_with_words(prompts: list[str], words_to_filter: set[str]) -> list[str]:
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

    prompts_before = len(prompts)
    prompts = [
        prompt
        for prompt in prompts
        if 5 <= len(prompt) <= 100 and not any(word in words_to_filter for word in prompt.lower().split())
    ]

    logger.info(f"Filtered undesired words. Prompts before: {prompts_before}. Prompts after: {len(prompts)}")

    return prompts


def correct_non_finished_prompts(prompts: list[str], prepositions: set[str]) -> list[str]:
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

    filtered_prompts = []
    num_augmented = 0

    for prompt in prompts:
        words = prompt.strip().split()
        if words and words[-1].lower() in prepositions:
            filtered_prompts.append(" ".join(words[:-1]))
            num_augmented += 1
        else:
            filtered_prompts.append(prompt)

    logger.info(f"Corrected non-finished prompts. {num_augmented} out of {len(prompts)} prompts")

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

    pattern = re.compile(r"[^a-zA-Z`\s-]")
    result_prompts = []
    for prompt in prompts_list:
        for line in prompt.split("\n"):
            line = pattern.sub("", line).strip().lower()
            if len(line.split()) > 3:
                result_prompts.append(line)
    return result_prompts


def remove_words_from_prompts(prompts: list[str], words_to_remove: set[str]) -> list[str]:
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
    num_augmented = 0

    for prompt in prompts:
        original = prompt
        filtered = " ".join(word for word in prompt.split() if word not in words_to_remove)
        result_prompts.append(filtered)

        if original != filtered:
            num_augmented += 1

    logger.info(f"Removed undesired words. {num_augmented} out of {len(prompts)} prompts")

    return result_prompts
