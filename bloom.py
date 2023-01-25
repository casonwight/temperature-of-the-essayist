from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers import set_seed


def download_model_online(model_name: str, model_path: str):
    model = BloomForCausalLM.from_pretrained(model_name)
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def load_model(model_path: str):
    model = BloomForCausalLM.from_pretrained(model_path)
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)

    return model, tokenizer


def run_model(prompt, prompt_short, actual_output, 
              model=None, tokenizer=None, 
              temperatures=None,
              verbose=False, **kwargs):
    set_seed(4242)
    if temperatures is None:
        raise ValueError("Temperatures must be specified")
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model("models/bloom-1b1")

    if "max_new_tokens" not in kwargs.keys():
        tokenized_actual_output = tokenizer(actual_output, return_tensors='pt')
        num_tokens_actual_output = tokenized_actual_output['input_ids'].shape[1]
        result_length = int(3 * num_tokens_actual_output)
    else:
        result_length = kwargs['max_new_tokens']

    model_params = {
        'early_stopping': True,
        'do_sample': True,
        'no_repeat_ngram_size': 2,
        'max_new_tokens': result_length
    }
    model_params.update(kwargs)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    if verbose:
        print("-" * 80)
        print("-" * 80)
        print("Prompt: ", prompt_short)
        print("-" * 80)
        print("Actual Output: ", actual_output)
        print("-" * 80)

    pred_outputs = []
    for temp in temperatures:
        model_params['temperature'] = temp
        output = model.generate(input_ids, **model_params)
        pred_output = tokenizer.decode(output[0], skip_special_tokens=True)
        end_of_prompt = prompt[-100:]
        pred_output_short = pred_output[(pred_output.index(end_of_prompt) + len(end_of_prompt)):]
        pred_outputs.append(pred_output_short)
        if verbose:
            print(f"Predicted Output (w/temp {temp:.2f}): ", pred_output_short)
            print("-" * 80)
    
    if verbose:
        print("-" * 80)

    return pred_outputs


if __name__=="__main__": 
    # download_model_online("bigscience/bloom-1b1", "models/bloom-1b1")

    model, tokenizer = load_model("models/bloom-1b1")

    run_model("Last Saturday, my friend annoyed me because he ",
              "Last Saturday, my friend annoyed me because he ",
              "wasted all his lunch money.",
              model=model, tokenizer=tokenizer,
              temperatures=[0.5, 0.7, 1.0, 1.2, 1.5],
              verbose=True)