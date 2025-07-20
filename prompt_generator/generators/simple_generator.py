import random as rd
import string
import tqdm
from pydantic import BaseModel

from prompt_generator.generators import AbstractGenerator

class Output(BaseModel):
    category: str
    object_name: str
    description: str

class SimpleGenerator(AbstractGenerator):
    DEFAULT_TEMPLATE = "random_letters"
    
    def generate(self) -> list[str]:
        output_prompts = []
        weights = [1.0] * 26
        weights[16] = 0.1 # Q
        weights[23] = 0.1 # X
        weights[24] = 0.1 # Y
        weights[25] = 0.1 # Z

        for i in tqdm.tqdm(range(self._number_of_prompts)):
            category_letter = rd.choices(string.ascii_uppercase, weights=weights)
            object_letter = rd.choices(string.ascii_uppercase, weights=weights)
            instruction_prompt = self._instruction_template.render(category_letter=category_letter, object_letter=object_letter)
            output = Output.model_validate_json(self._backend.generate(instruction_prompt, structured_output=Output.model_json_schema()))
            output_prompts.append(output.description)

        return output_prompts