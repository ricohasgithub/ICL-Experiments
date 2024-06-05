
import sys

from train import Trainer
from datasets.dataset import SeqGenerator

from models.transformer import Transformer
from models.embedding import InputEmbedder

def experiment_base():

    input_embedding = InputEmbedder(example_encoding="linear")
    model = Transformer(input_embedder=input_embedding)
    
    seq_generator_factory = SeqGenerator()
    data_generator = seq_generator_factory.get_bursty_seq

    trainer = Trainer(model, data_generator)
    trainer.train()

if __name__ == "__main__":

    # Read command line inputs to decide which experiment to run
    experiment_id = sys.argv[1]

    if experiment_id == "base":
        experiment_base()
    elif experiment_id == "mixed":
        pass