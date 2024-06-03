
from train import Trainer
from datasets.dataset import SeqGenerator

from models.transformer import Transformer
from models.embedding import InputEmbedder

def experiment_base():

    input_embedding = InputEmbedder(example="resnet")
    model = Transformer(input_embedder=input_embedding)
    
    seq_generator_factory = SeqGenerator()
    data_generator = seq_generator_factory.get_bursty_seq

    trainer = Trainer(model, data_generator)
    trainer.train()