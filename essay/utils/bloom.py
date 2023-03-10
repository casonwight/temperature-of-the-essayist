from transformers import BloomForCausalLM, BloomTokenizerFast, set_seed
import torch
from torch.nn import functional as F


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


def generate_probabilities(prompt, 
                           model=None, tokenizer=None, 
                           verbose=False, 
                           temperature=1.0,
                           output_length=1,
                           num_outputs=10,
                           model_path="essay/models/bloom-1b1"):
    """ 
    Generate probabilities for top next predicted words using a Bloom model 
    """
    set_seed(4242)

    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids

    with torch.no_grad():
        # We want 5 options for only the next word
        # The output is of the class BeamSearchDecoderOnlyOutput  
        beam_search_output = model.generate(
            inputs=input_ids,
            max_new_tokens=output_length,
            num_beams=num_outputs,
            num_return_sequences=num_outputs,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=temperature
        )

    # Decode the 5 output sequences
    output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in beam_search_output.sequences]
    
    # Get the probabilities for each
    probs = F.softmax(beam_search_output.sequences_scores, dim=0).numpy()

    outputs = [(out_text, prob) for out_text, prob in zip(output_texts, probs)]

    if verbose:
        print(f"Prompt: {prompt}")
        for i, (out_text, prob) in enumerate(outputs):
            print(f"{i+1} (probability of {100*prob:.2f}%): {out_text.replace(prompt, '')}")

    return outputs


def calculate_perplexity(prompt, 
                         model=None, tokenizer=None, 
                         stride=1,
                         model_path="essay/models/bloom-1b1",
                         start_calculating_at=0):
    set_seed(4242)

    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(start_calculating_at, seq_len, stride):
        end_loc = seq_len
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()

    return ppl
    


if __name__=="__main__": 
    # download_model_online("bigscience/bloom-1b1", "models/bloom-1b1")
    # print(calculate_perplexity("To be or not to be: that is the question"))
    # print(calculate_perplexity("To be or not to be: that is the poop stain"))
    # print(calculate_perplexity("I like dogs. I like cats. I like all kinds of things."))

    prompt = """When I was in fourth grade, my class took a field trip to the American Tobacco plant in nearby Durham, North Carolina. There we witnessed the making of cigarettes and were given free packs to take home to our parents. I tell people this and they ask me how old I am, thinking, I guess, that I went to the world's first elementary school, one where we wrote on cave walls and hunted our lunch with clubs. Then I mention the smoking lounge at my high school. It was outdoors, but, still, you'd never find anything like that now, not even if the school was in a prison.

I recall seeing ashtrays in movie theatres and grocery stores, but they didn't make me want to smoke. In fact, it was just the opposite. Once, I drove an embroidery needle into my mother's carton of Winstons, over and over, as if it were a voodoo doll. She then beat me for twenty seconds, at which point she ran out of breath and stood there panting, ???????That's . . . not . . . funny.???????

A few years later, we were sitting around the breakfast table and she invited me to take a puff. I did. Then I ran to the kitchen and drained a carton of orange juice, drinking so furiously that half of it ran down my chin and onto my shirt. How could she, or anyone, really, make a habit of something so fundamentally unpleasant? When my sister Lisa started smoking, I forbade her to enter my bedroom with a lit cigarette. She could talk to me, but only from the other side of the threshold, and she had to avert her head when she exhaled. I did the same when my sister Gretchen started.

It wasn't the smoke but the smell of it that bothered me. In later years, I didn't care so much, but at the time I found it depressing: the scent of neglect. It wasn't so noticeable in the rest of the house, but then again the rest of the house was neglected. My room was clean and orderly, and if I'd had my way it would have smelled like an album jacket the moment you remove the plastic. That is to say, it would have smelled like"
"""

    act_output = """ anticipation."""

    model, tokenizer = load_model("essay/models/bloom-1b1")
    print(calculate_perplexity(
        prompt + act_output,
        start_calculating_at=len(tokenizer(prompt, return_tensors='pt').input_ids)
    ))
