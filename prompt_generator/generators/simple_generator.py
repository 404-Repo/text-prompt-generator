import random as rd
import string
import tqdm

from prompt_generator.generators import AbstractGenerator

class SimpleGenerator(AbstractGenerator):
    DEFAULT_TEMPLATE = "random_letters"
    
    def generate(self) -> list[str]:
        output_prompts = []
        weights = [1.0] * 26
        weights[16] = 0.2 # Q
        weights[23] = 0.2 # X
        weights[24] = 0.2 # Y

        for i in tqdm.tqdm(range(self._number_of_prompts)):
            category_letter = rd.choices(string.ascii_uppercase, weights=weights)
            object_letter = rd.choices(string.ascii_uppercase, weights=weights)
            instruction_prompt = self._instruction_template.render(category_letter=category_letter, object_letter=object_letter)
            output_prompts.append(self._backend.generate(instruction_prompt))

        return output_prompts