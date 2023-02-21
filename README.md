# Temperature of the Essayist

The purpose of this repo is to explore modern essays through the lens of a modern large language model (LLM). 
To use AMD GPU, see [this guide](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows).


## Model
The model used here is [BLOOM](https://huggingface.co/bigscience/bloom). 
BLOOM is an open-source model, trained on nearly half a trillion tokens across nearly 50 languages.
BLOOM has several sizes, including 
Due to resource constraints, this project uses the 1.1B-parameter version.


## Essays
### data/essay-1
- Brenda Miller
- Shape of Emptiness 
- what is the next word? 
- what if we let it write sentence? 
- this essay is really short (750 words) 
	
	
### data/essay-2
- David Sedaris
- Letting Go
- There are two different places I want us to try. I want to see what it will put in for album jacket and then a later fill in for anticipation. When you fill in anticipation, leave album jacket as it appears in the essay. 
- For both we should try to let it write the next word
- Then let it write the next sentence
- Then let it write the next paragraph
	
### data/essay-3
- Anne Dilliard
- Total Eclipse 
- let the model finish the phrase


## To Do
- [x] Get essays
- [x] Autocomplete results
- [ ] Probability word trees (w/comparisons to actual)
- [x] Perplexity (temperatures compared to actual)
- [x] Animated bar chart of temperature changing temperatures
- [x] Animation explaining transformers
- [x] Powerpoint slide outline
- [ ] Generate extended output at authors' temperature