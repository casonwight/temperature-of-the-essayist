from utils.bloom import load_model, generate_text
import numpy as np
import pandas as pd
import re


MIN_TEMP = 0.1
MAX_TEMP = 2.0
NUM_TEMPS = 20
NUM_ESSAYS = 4
LINEBREAK = "-" * 80 + "\n"

class AutocompleteExperiment:
    def __init__(self):
        self.model, self.tokenizer = load_model("essay/models/bloom-1b1")
        self.temperatures = np.linspace(MIN_TEMP, MAX_TEMP, NUM_TEMPS)
        self.essay_nums = range(1, NUM_ESSAYS+1)
        self.essays = [self.read_essay(f"essay/data/essay-{i}.txt") for i in self.essay_nums]

    @staticmethod
    def extract_prompt(essay, change, start=0):
        prompt = (
            essay[start:change]
            .replace('[START]', '')
            .replace('[CHANGEPOINT]', '')
            .replace('[END]', '')
        )
           
        return prompt

    def read_essay(self, filepath):
        with open(filepath, 'r', encoding="utf8") as f:
            essay = f.read()

        start_indices = [i.start() for i in re.finditer('\[START\]', essay)]
        change_indices = [i.start() for i in re.finditer('\[CHANGEPOINT\]', essay)]
        end_indices = [i.start() for i in re.finditer('\[END\]', essay)]

        if len(start_indices) != len(change_indices) or len(change_indices) != len(end_indices):
            raise ValueError("Number of start, change, and end indices must be equal")

        prompts = [self.extract_prompt(essay, change) for change in change_indices]
        prompts_short = [self.extract_prompt(essay, change, start) for change, start in zip(change_indices, start_indices)]
        actual_outputs = [self.extract_prompt(essay, end, change) for end, change in zip(end_indices, change_indices)]

        output = {
            'prompts': prompts,
            'prompts_short': prompts_short,
            'actual_outputs': actual_outputs
        }

        return output


    def run(self, verbose=True, **kwargs):
        results_df = pd.DataFrame(columns=['essay', 'full_prompt', 'short_prompt', 'act_output', 'temp', 'pred_output'])

        for i, essay in zip(self.essay_nums, self.essays):
            if verbose:
                print(f"Running essay {i+1}...")
            prompts = essay['prompts']
            prompts_short = essay['prompts_short']
            actual_outputs = essay['actual_outputs']

            with open(f"essay/results/autocomplete-results-essay-{i+1}.txt", 'w') as f:
                for prompt, prompt_short, actual_output in zip(prompts, prompts_short, actual_outputs):
                    pred_outputs = generate_text(
                        prompt, prompt_short, actual_output, 
                        model=self.model, tokenizer=self.tokenizer, 
                        temperatures=self.temperatures,
                        verbose=verbose, **kwargs
                    )
                    f.write(LINEBREAK + LINEBREAK)
                    f.write(f"Prompt: {prompt_short}\n")
                    f.write(LINEBREAK)
                    f.write(f"Actual Output: {actual_output}\n")
                    f.write(LINEBREAK)
                    for pred_output, temp in zip(pred_outputs, self.temperatures):
                        f.write(f"Predicted Output (w/temp {temp}): {pred_output}\n")
                        f.write(LINEBREAK)
                    f.write(LINEBREAK + "\n\n")

                    prompt_results = pd.DataFrame({
                        'essay': [i] * len(self.temperatures), 
                        'full_prompt': [prompt] * len(self.temperatures), 
                        'short_prompt': [prompt_short] * len(self.temperatures), 
                        'act_output': [actual_output] * len(self.temperatures), 
                        'temp': self.temperatures, 
                        'pred_output': pred_outputs
                    })
                    results_df = results_df.append(prompt_results, ignore_index=True)
        results_df.to_csv("essay/results/autocomplete-results.csv")



if __name__=="__main__":
    experiment = AutocompleteExperiment()
    experiment.run()
