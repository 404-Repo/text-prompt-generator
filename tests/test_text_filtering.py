import pytest
from textblob import TextBlob

from symspellpy import SymSpell
import pkg_resources
from itertools import islice


def test_list_to_set():
    test_dataset = ["plants",
                    "a building with red wall frame",
                    "pink flamingo",
                    "A gold Turtle",
                    "a gold turtle",
                    "Earrings with gemstones",
                    "a building with yellow column base",
                    "a Building with Yellow Column Base",
                    "a bronze gorilla"]

    res_lines = []
    for l in test_dataset:
        res_lines.append(' '.join(word.lower() for word in l.split()))

    res_lines = list(set(res_lines)).sort()

    ref_lines = ["plants",
                 "a building with red wall frame",
                 "pink flamingo",
                 "a gold turtle",
                 "Earrings with gemstones",
                 "a building with yellow column base",
                 "a bronze gorilla"].sort()

    assert ref_lines == res_lines


def test_filter_articles():
    test_dataset = ["Plants",
                    "a building with red wall frame",
                    "A building with the red wall frame",
                    "pink flamingo",
                    "A gold Turtle",
                    "the gold turtle",
                    "Earrings with Gemstones",
                    "building with the yellow column base",
                    "bronze gorilla",
                    "An elephant in the zoo"]

    res_lines = []
    for l in test_dataset:
        res_lines.append(' '.join(word.lower() for word in l.split()))

    articles = ["a", "the", "an"]
    res_lines = [' '.join(word for word in sentence.split() if word.lower() not in articles) for sentence in res_lines].sort()

    ref_lines = ["Plants",
                 "building with red wall frame",
                 "building with red wall frame",
                 "pink flamingo",
                 "gold turtle",
                 "gold turtle",
                 "earrings with gemstones",
                 "building with yellow column base",
                 "bronze gorilla",
                 "elephant in zoo"].sort()

    assert ref_lines == res_lines


def test_filter_short_strings():
    test_data = ["I ab", "I c", "red apple", "cat", "field", "bc"]
    res_lines = list(filter(lambda sentence: 3 <= len(sentence) <= float("inf"), test_data)).sort()

    ref_lines = ["red apple", "field"].sort()
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

    res_lines = []
    for l in test_dataset:
        res_lines.append(' '.join(word.lower() for word in l.split()))

    res_lines = list(filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(filtering_words), res_lines)).sort()

    ref_lines = ["a building with red wall frame",
                 "the gold turtle",
                 "earrings with gemstones",
                 "building with the yellow column base"].sort()

    assert ref_lines == res_lines


def test_single_colour_filter():
    single_word_colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "grey"]
    test_dataset = ["A golden frog",
                    "black",
                    "frog",
                    "Green bug",
                    " green",
                    "car"]
    res_lines = []
    for l in test_dataset:
        res_lines.append(' '.join(word.lower() for word in l.split()))

    res_lines = [l for l in res_lines if l not in single_word_colors].sort()

    ref_lines = ["a golden frog",
                 "frog",
                 "green bug",
                 "car"].sort()

    assert ref_lines == res_lines


def test_grammar_check():
    test_dataset = ["malachite earring with gold hook",
                    "seahorse with blue and orange striped tail",
                    "checkered lego table with chairs",
                    "white hospital building",
                    "squirrel with puple fur and red tail"]

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
    )

    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    corrected_prompts = []
    for p in test_dataset:
        terms = sym_spell.lookup_compound(p, max_edit_distance=2)
        corrected_prompts.append(terms[0].term)

    ref_prompts = ["malachite earring with gold hook",
                   "seahorse with blue and orange striped tail",
                   "check red lego table with chairs",
                   "white hospital building",
                   "squirrel with purple fur and red tail"]

    assert ref_prompts == corrected_prompts
