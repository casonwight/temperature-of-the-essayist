from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers import set_seed
import torch


LINEBREAK = "-" * 80
MAX_LENGTH_COEFF = 3
MAX_REPETITIONS = 2
PROMPT_END_LENGTH = 100


def download_model_online(model_name: str, model_path: str):
    """ Download model from HuggingFace and save locally """
    model = BloomForCausalLM.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def load_model(model_path: str):
    """ Load model from local path """
    model = BloomForCausalLM.from_pretrained(model_path)
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    return model, tokenizer


def generate_text(prompt, prompt_short, actual_output, 
                  model=None, tokenizer=None, 
                  temperatures=None,
                  verbose=False, 
                  model_path="essay/models/bloom-1b1", 
                  **kwargs):
    """ Generate text using a Bloom model, at various temperatures """
    set_seed(4242)
    if temperatures is None:
        raise ValueError("Temperatures must be specified")
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)

    if "max_new_tokens" not in kwargs.keys():
        tokenized_actual_output = tokenizer(actual_output, return_tensors='pt')
        num_tokens_actual_output = tokenized_actual_output['input_ids'].shape[1]
        result_length = int(MAX_LENGTH_COEFF * num_tokens_actual_output)
    else:
        result_length = kwargs['max_new_tokens']

    model_params = {
        'early_stopping': True,
        'do_sample': True,
        'no_repeat_ngram_size': MAX_REPETITIONS,
        'max_new_tokens': result_length
    }
    model_params.update(kwargs)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    if verbose:
        print(LINEBREAK)
        print(LINEBREAK)
        print("Prompt: ", prompt_short)
        print(LINEBREAK)
        print("Actual Output: ", actual_output)
        print(LINEBREAK)

    pred_outputs = []
    for temp in temperatures:
        model_params['temperature'] = temp
        output = model.generate(input_ids, **model_params)
        pred_output = tokenizer.decode(output[0], skip_special_tokens=True)
        end_of_prompt = prompt[-PROMPT_END_LENGTH:]
        pred_output_short = pred_output[(pred_output.index(end_of_prompt) + len(end_of_prompt)):]
        pred_outputs.append(pred_output_short)
        if verbose:
            print(f"Predicted Output (w/temp {temp:.2f}): ", pred_output_short)
            print(LINEBREAK)
    
    if verbose:
        print(LINEBREAK)

    return pred_outputs


def generate_probabilities(prompt, prompt_short, actual_output, 
                           model=None, tokenizer=None, 
                           verbose=False, 
                           model_path="essay/models/bloom-1b1"):
    """ Generate probabilities for top next predicted words using a Bloom model 
    The result of this function is a tree of probabilities for the next word,
    given the previous words. 

    The tree is represented as a dictionary, where the keys are the next words,
    and the values are the probabilities of those words. The values can be 
    either floats (if the word is the last word in the sequence), or another
    dictionary (if the word is not the last word in the sequence).

    For example, if the next word is "the", and the next word after that is "cat",
    then the dictionary will look like this:
        {"the": {"cat": 0.5}}

    We will only keep the 10 total most probable paths (leaf nodes).

    The temperature is set to 1.0 by default.
    """
    set_seed(4242)

    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)

    model_params = {
        'early_stopping': True,
        'do_sample': True,
        'no_repeat_ngram_size': MAX_REPETITIONS,
        'max_new_tokens': 1,
        'temperature': 1.0
    }
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    if verbose:
        print(LINEBREAK)
        print(LINEBREAK)
        print("Prompt: ", prompt_short)
        print(LINEBREAK)
        print("Actual Output: ", actual_output)
        print(LINEBREAK)

    output = model.generate(input_ids, **model_params)
    pred_output = tokenizer.decode(output[0], skip_special_tokens=True)
    end_of_prompt = prompt[-PROMPT_END_LENGTH:]
    pred_output_short = pred_output[(pred_output.index(end_of_prompt) + len(end_of_prompt)):]
    if verbose:
        print(f"Predicted Output: ", pred_output_short)
        print(LINEBREAK)
        print(LINEBREAK)

    # Get the top 10 most probable paths
    probs = model(input_ids, return_dict=True).logits[0]
    
    print(probs)
    print(probs.shape)

    return probs






if __name__=="__main__": 
    # download_model_online("bigscience/bloom-1b1", "models/bloom-1b1")

    model, tokenizer = load_model("essay/models/bloom-1b1")

    generate_probabilities(
        prompt="Last Saturday, my friend annoyed me because he ",
        prompt_short="Last Saturday, my friend annoyed me because he ",
        actual_output="wasted all his lunch money.",
        model=model, 
        tokenizer=tokenizer,
        verbose=True
    )