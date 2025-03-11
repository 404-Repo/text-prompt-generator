import generator.utils.prompts_filtering_utils as prompt_filters


def test_unique_prompts_filtering():
    test_dataset = [
        "plants",
        "a building with red wall frame",
        "pink flamingo",
        "A gold Turtle",
        "a gold turtle",
        "Earrings with gemstones",
        "a building with yellow column base",
        "a Building with Yellow Column Base",
        "a bronze gorilla",
    ]

    res_lines = prompt_filters.filter_unique_prompts(test_dataset).sort()

    ref_lines = [
        "plants",
        "a building with red wall frame",
        "pink flamingo",
        "a gold turtle",
        "Earrings with gemstones",
        "a building with yellow column base",
        "a bronze gorilla",
    ].sort()

    assert ref_lines == res_lines


def test_filter_prompts_with_words():
    filtering_words = {"water", "ocean", "lake", "zoo", "field"}
    test_dataset = [
        "Plants in the field",
        "a building with red wall frame",
        "pink flamingo standing in the lake",
        "A gold Turtle swimming in the Ocean",
        "the gold turtle",
        "Earrings with Gemstones",
        "building with the yellow column base",
        "bronze gorilla drinking a water from the glass",
        "An elephant in the zoo",
    ]

    res_lines = prompt_filters.filter_prompts_with_words(test_dataset, filtering_words).sort()

    ref_lines = [
        "a building with red wall frame",
        "the gold turtle",
        "earrings with gemstones",
        "building with the yellow column base",
    ].sort()

    assert ref_lines == res_lines


def test_correction_of_non_finished_prompts():
    test_dataset = [
        "female android with",
        "black slanted game controller on",
        "viking warrior holding axe by",
        "dark green horned griffin at",
        "dark chocolate covered raisins with",
        "black and white bone-clawed manticore about",
        "giraffe with long neck reaching leaves against",
        "curved lounge chair in orange among",
        "green triangular drone propeller behind",
        "red angled tablet stand before",
        "red and blue segmented snake-like robot between",
        "orange tabby cat washing its face during",
        "brown bear standing on hind legs for",
        "majestic redwood tree trunk from",
        "gold elongated vr controller of",
        "dark oak barrel chair to",
        "twisting steel and glass skyscraper over",
        "red and black striped chimera under",
        "melting vanilla ice cream cone through",
        "blue and green humanoid robot with crossed arms into",
        "cybernetic ninja in black suit upon",
        "black and gold ottoman within",
        "pineapple upside down cake without",
        "purple wired gaming mouse along",
        "green spider-like robot with outstretched limbs across",
        "blue viking war hammer behind",
        "brown twisted tree bark beneath",
        "silver space-age laser gun beside",
        "android knight with lance beyond",
        "white polar bear hunting for seals near",
        "blue circular smartwatch face off",
        "cyberpunk punk with leather jacket onto",
        "steampunk inventor in goggles towards",
        "gold twisted headphone cable underneath",
    ]

    prepositions = {
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
    }

    res_lines = prompt_filters.correct_non_finished_prompts(test_dataset, prepositions)

    ref_lines = [
        "female android",
        "black slanted game controller",
        "viking warrior holding axe",
        "dark green horned griffin",
        "dark chocolate covered raisins",
        "black and white bone-clawed manticore",
        "giraffe with long neck reaching leaves",
        "curved lounge chair in orange",
        "green triangular drone propeller",
        "red angled tablet stand",
        "red and blue segmented snake-like robot",
        "orange tabby cat washing its face",
        "brown bear standing on hind legs",
        "majestic redwood tree trunk",
        "gold elongated vr controller",
        "dark oak barrel chair",
        "twisting steel and glass skyscraper",
        "red and black striped chimera",
        "melting vanilla ice cream cone",
        "blue and green humanoid robot with crossed arms",
        "cybernetic ninja in black suit",
        "black and gold ottoman",
        "pineapple upside down cake",
        "purple wired gaming mouse",
        "green spider-like robot with outstretched limbs",
        "blue viking war hammer",
        "brown twisted tree bark",
        "silver space-age laser gun",
        "android knight with lance",
        "white polar bear hunting for seals",
        "blue circular smartwatch face",
        "cyberpunk punk with leather jacket",
        "steampunk inventor in goggles",
        "gold twisted headphone cable",
    ]

    assert ref_lines == res_lines
