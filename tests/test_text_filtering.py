import generator.utils.prompts_filtering_utils as prompt_filters


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

    res_lines = prompt_filters.filter_unique_prompts(test_dataset).sort()

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

    res_lines = prompt_filters.filter_prompts_with_words(test_dataset, filtering_words).sort()

    ref_lines = ["a building with red wall frame",
                 "the gold turtle",
                 "earrings with gemstones",
                 "building with the yellow column base"].sort()

    assert ref_lines == res_lines


def test_correction_of_non_finished_prompts():
    test_dataset = ["female android with",
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
                    "gold twisted headphone cable underneath"]

    res_lines = prompt_filters.correct_non_finished_prompts(test_dataset)

    ref_lines = ["female android\n",
                 "black slanted game controller\n",
                 "viking warrior holding axe\n",
                 "dark green horned griffin\n",
                 "dark chocolate covered raisins\n",
                 "black and white bone-clawed manticore\n",
                 "giraffe with long neck reaching leaves\n",
                 "curved lounge chair in orange\n",
                 "green triangular drone propeller\n",
                 "red angled tablet stand\n",
                 "red and blue segmented snake-like robot\n",
                 "orange tabby cat washing its face\n",
                 "brown bear standing on hind legs\n",
                 "majestic redwood tree trunk\n",
                 "gold elongated vr controller\n",
                 "dark oak barrel chair\n",
                 "twisting steel and glass skyscraper\n",
                 "red and black striped chimera\n",
                 "melting vanilla ice cream cone\n",
                 "blue and green humanoid robot with crossed arms\n",
                 "cybernetic ninja in black suit\n",
                 "black and gold ottoman\n",
                 "pineapple upside down cake\n",
                 "purple wired gaming mouse\n",
                 "green spider-like robot with outstretched limbs\n",
                 "blue viking war hammer\n",
                 "brown twisted tree bark\n",
                 "silver space-age laser gun\n",
                 "android knight with lance\n",
                 "white polar bear hunting for seals\n",
                 "blue circular smartwatch face\n",
                 "cyberpunk punk with leather jacket\n",
                 "steampunk inventor in goggles\n",
                 "gold twisted headphone cable\n"]

    assert ref_lines == res_lines
