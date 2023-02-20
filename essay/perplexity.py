from utils.bloom import load_model, calculate_perplexity
import pandas as pd


class PerplexityExperiment:
    def __init__(self):
        self.model, self.tokenizer = load_model("essay/models/bloom-1b1")
        self.generated_outputs = pd.read_csv("essay/results/autocomplete-results.csv").drop(columns=['Unnamed: 0'])
        self.generated_outputs['perplexity'] = [None for _ in range(len(self.generated_outputs))]   

    def run(self, verbose=True, **kwargs):
        for i in self.generated_outputs.index:
            if verbose:
                essay = self.generated_outputs.loc[i, 'essay']
                temp = self.generated_outputs.loc[i, 'temp']
                print(f"Calculating perplexity for essay {essay+1} temp {temp} predicted outputs...")
            prompt = self.generated_outputs.loc[i, 'full_prompt']
            pred_output = self.generated_outputs.loc[i, 'pred_output']

            prompt_length = len(self.tokenizer(prompt, return_tensors='pt').input_ids)
            full_text = prompt + pred_output

            self.generated_outputs.loc[i, 'perplexity'] = calculate_perplexity(
                full_text, 
                model=self.model, tokenizer=self.tokenizer,
                start_calculating_at=prompt_length,
                **kwargs
            )
        
        for i in self.generated_outputs['essay'].unique():
            if verbose:
                print(f"Calculating perplexity for essay {i+1} actual outputs...")
            prompt = self.generated_outputs.query(f"essay == {i}").iloc[0]['full_prompt']
            short_prompt = self.generated_outputs.query(f"essay == {i}").iloc[0]['short_prompt']
            actual_output = self.generated_outputs.query(f"essay == {i}").iloc[0]['act_output']

            prompt_length = len(self.tokenizer(prompt, return_tensors='pt').input_ids)
            full_text = prompt + actual_output

            perplexity = calculate_perplexity(
                full_text, 
                model=self.model, tokenizer=self.tokenizer,
                start_calculating_at=prompt_length,
                **kwargs
            )

            self.generated_outputs = self.generated_outputs.append(
                {
                    'essay': i,
                    'full_prompt': prompt,
                    'short_prompt': short_prompt,
                    'act_output': actual_output,
                    'temp': None,
                    'pred_output': None,
                    'perplexity': perplexity
                }, ignore_index=True
            )

            formatted_outputs = (
                self.generated_outputs
                .sort_values(by=['essay', 'temp'])
            )
                
            formatted_outputs.to_csv("essay/results/perplexity-results.csv")

if __name__ == "__main__":
    experiment = PerplexityExperiment()
    experiment.run()