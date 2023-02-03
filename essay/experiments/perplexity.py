from ..utils.bloom import load_model, generate_text


NUM_ESSAYS = 3

class PerplexityExperiment:
    def __init__(self):
        self.model, self.tokenizer = load_model("essay/models/bloom-1b1")
        self.essays = [self.read_essay(f"essay/data/essay-{i}.txt") for i in range(1, NUM_ESSAYS+1)]

    def run(self, verbose=True, **kwargs):
        pass