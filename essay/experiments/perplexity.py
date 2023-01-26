from ..utils.bloom import load_model, run_model

class PerplexityExperiment:
    def __init__(self):
        self.model, self.tokenizer = load_model("essay/models/bloom-1b1")


    def run(self, verbose=True, **kwargs):
        pass