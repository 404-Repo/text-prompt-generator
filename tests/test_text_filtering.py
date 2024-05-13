from Generator.prompt_checker import PromptChecker
from Generator.utils import load_config_file
import pytest
import torch


def test_unique_prompts_filtering():
    test_dataset = ["plants",
                    "a building with red wall frame",
                    "pink flamingo",
                    "A gold Turtle",
                    "a gold turtle",
                    "Earrings with gemstones",
                    "a building with yellow column base",
                    "a Building with Yellow Column Base",
                    "a bronze gorilla"]

    data_config = load_config_file()
    checker = PromptChecker(data_config)
    res_lines = checker.filter_unique_prompts(test_dataset).sort()

    ref_lines = ["plants",
                 "a building with red wall frame",
                 "pink flamingo",
                 "a gold turtle",
                 "Earrings with gemstones",
                 "a building with yellow column base",
                 "a bronze gorilla"].sort()

    assert ref_lines == res_lines


def test_filter_prompts_with_words():
    filtering_words = ["water", "ocean", "lake", "zoo", "field"]
    test_dataset = ["Plants in the field",
                    "a building with red wall frame",
                    "pink flamingo standing in the lake",
                    "A gold Turtle swimming in the Ocean",
                    "the gold turtle",
                    "Earrings with Gemstones",
                    "building with the yellow column base",
                    "bronze gorilla drinking a water from the glass",
                    "An elephant in the zoo"]

    data_config = load_config_file()
    checker = PromptChecker(data_config)
    res_lines = checker.filter_prompts_with_words(test_dataset, filtering_words).sort()

    ref_lines = ["a building with red wall frame",
                 "the gold turtle",
                 "earrings with gemstones",
                 "building with the yellow column base"].sort()

    assert ref_lines == res_lines


data_config = load_config_file()
checker = PromptChecker(data_config)
checker.preload_vllm_model()


def test_prompt_quality_check():
    prompts = ["white fan shaped bird of paradise flower",
               "a brown bear",
               "super spider shaped flower in vacuum",
               "sleeping cat on the bed"]

    wrong_prompts = []
    correct_prompts = []
    for p in prompts:
        score = checker.vllm_check_prompt(p)

        if float(score) >= 0.5:
            correct_prompts.append(p)
        else:
            wrong_prompts.append(p)

    assert "a brown bear" in correct_prompts
    assert "sleeping cat on the bed" in correct_prompts
    assert "super spider shaped flower in vacuum" in wrong_prompts
    assert "white fan shaped bird of paradise flower" in wrong_prompts


def test_prompt_correction():
    prompts = ["white fan shaped bird of paradise flower",
               "a brown bear",
               "super spider shaped flower in vacuum",
               "sleeping cat on the bed"]

    corrected_prompts = []
    for p in prompts:
        score = checker.vllm_check_prompt(p)

        if float(score) >= 0.5:
            corrected_prompts.append(p)
        else:
            prompt = checker.vllm_correct_prompt(p)
            corrected_prompts.append(prompt)

    assert prompts[0] != corrected_prompts[0] and corrected_prompts[0] != ""
    assert prompts[1] == corrected_prompts[1] and corrected_prompts[1] != ""
    assert prompts[2] != corrected_prompts[2] and corrected_prompts[2] != ""
    assert prompts[3] == corrected_prompts[3] and corrected_prompts[3] != ""

# def test_checker_vllm_model_unload():
#     data_config = load_config_file()
#     checker = PromptChecker(data_config)
#     checker.preload_vllm_model()
#
#     gpu_cache, gpu_mem_alloc = checker.unload_vllm_model()
#
#     gpu_cache_after = torch.cuda.memory_cached()
#     gpu_mem_alloc_after = torch.cuda.memory_allocated()
#
#     assert gpu_cache_after < 1e+7
#     assert gpu_mem_alloc_after == 0
