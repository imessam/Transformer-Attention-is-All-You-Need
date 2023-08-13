# Transformer (Attention is All You Need) paper implementation from Scratch.

- Implemented using PyTorch Framework.
- Trained and tested on WMT14 EN-DE dataset.
- Trained and validated using only 30K sentences of the train set due to memory limitations.
- Tested on the whole test set.
- Using the same model architechture as the base model expect that the number of encoders and decoders layers are 1 not 6 as stated on the paper due to memory limitations.
- The notebooks are working notebooks both on a local machine or on Google Colab.
- The Transformer.ipynb notebook use simple self made tokenizer to preprocess the input sentences (uses utils.py). 
- The Transformer-BytePairEncoding.ipynb notebook use Huggingface BPE tokenizer to preprocess the input sentences (uses utils2.py). 
- The model overfits the train set due to insuffecient training data, but is confirms that the model is working as it should.