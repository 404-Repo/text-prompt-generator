import re
from typing import List

from loguru import logger


def filter_unique_prompts(prompts: List[str]):
    """
    Function for filtering all duplicates from the input prompt list

    Parameters
    ----------
    prompts: a list with input prompts

    Returns
    -------
    prompts: a list with unique prompts
    """
    logger.info(f"\n")
    logger.info("*" * 40)
    logger.info(" *** Filtering unique prompts. ***")
    logger.info("*" * 40)
    logger.info(f"\n")

    for i, p in enumerate(prompts):
        prompts[i] = ' '.join(word.lower() for word in p.split())

    logger.info(f" Total lines in the dataset before: {len(prompts)}")

    articles = ["a", "the", "an"]
    prompts = [' '.join(word for word in sentence.split() if word.lower() not in articles) for sentence in prompts]
    prompts = list(set(prompts))
    prompts = [l + "\n" if "\n" not in l else l for l in prompts]

    logger.info(f" Total lines in the dataset after: {len(prompts)}")
    logger.info(f"\n")

    return prompts


def filter_prompts_with_words(prompts: List[str], words_to_filter: List[str]):
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

    logger.info(f"\n")
    logger.info("*" * 40)
    logger.info(" *** Filtering prompts with undesired words. ***")
    logger.info("*" * 40)
    logger.info(f"\n")

    logger.info(f" Total lines in the dataset before: {len(prompts)}")

    prompts = list(filter(lambda sentence: 5 <= len(sentence) <= 100, prompts))
    prompts = list(
        filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(words_to_filter), prompts))
    prompts = [l + "\n" if "\n" not in l else l for l in prompts]

    logger.info(f" Total lines in the dataset after: {len(prompts)}")
    logger.info(f"\n")

    return prompts


def correct_non_finished_prompts(prompts: List[str]):
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

    prepositions = [
        'in', 'on', 'at', 'by', 'with', 'about', 'against', 'among', 'before',
        'behind', 'between', 'during', 'for', 'from', 'of', 'to', 'over', 'under',
        'through', 'into', 'upon', 'within', 'without', 'along', 'across', 'behind',
        'beneath', 'beside', 'beyond', 'near', 'off', 'onto', 'towards', 'underneath',
        'outside'
    ]
    pattern = re.compile(r'\b(' + '|'.join(prepositions) + r')\b\s*$', re.IGNORECASE)
    filtered_prompts = [re.sub(pattern, '', prompt).strip() + "\n" for prompt in prompts]
    return filtered_prompts


def post_process_generated_prompts(prompts_list: List[str]):
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
    for el in prompts_list:
        lines = el.split("\n")
        processed_lines = []
        for i in range(len(lines)):
            line = re.sub(r'[^a-zA-Z`\s-]', '', lines[i])
            line = re.sub(r'\d+', '', line)
            line = line.replace(".", "")
            line = line.replace("- ", "")

            if len(line.split()) > 3:
                if "\n" not in line:
                    line += "\n"
                processed_lines += [line]
        result_prompts += processed_lines
    return result_prompts
