"""Dev code as I understand this."""

with open('data/shakespeare.txt', 'r') as file:
    text = file.read()

print(f"Text has {len(text)} characters")

# create the set of all characters in the text
chars = set(text)

# convert to list and sort
chars = sorted(list(chars))

print(f"Text has {len(chars)} unique characters")
print(chars)

# lets enumerate to create a dictionary, one from int to char, and one from char to int
int_to_str = dict(enumerate(chars))
str_to_int = {c: i for i, c in int_to_str.items()}

# printing the dictionaries
print(int_to_str)
print(str_to_int)

def encode_string(s: str, mapping: dict) -> list:
    return [mapping[c] for c in s]

def decode_string(l: list, mapping: dict) -> str:
    return ''.join([mapping[i] for i in l])

# lets encode and decode a string
encoded = encode_string("hello world, \nmy name is gpt", str_to_int)
print(encoded)

decoded = decode_string(encoded, int_to_str)
print(decoded)

# now we could encode the entirety of the text
encoded_text = encode_string(text, str_to_int)

# and decode it back
decoded_text = decode_string(encoded_text, int_to_str)

# printing the first 100 characters
print(text[:100])
print(decoded_text[:100])

import torch
data = torch.tensor(encoded_text, dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

split = int(0.9 * len(data))
train_data = data[:split]
test_data = data[split:]

torch.manual_seed(1337)
# context length = block_size
block_size = 8
batch_size = 4 # sequences in parallel

def get_batch(split: str):
    """Generate a small batch of data of inputs x and targets y.
    
    Takes the batch size 4, to create 4 random starting points in the data.
    Then creates 4 sequences of length 8, and 4 sequences of length 8 shifted by 1 for targets.
    """
    data = train_data if split == 'train' else test_data
    batch_start_indeces = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in batch_start_indeces])
    y = torch.stack([data[i+1:i+block_size+1] for i in batch_start_indeces])
    return x, y

xb, yb = get_batch('train')
print(f"Input shape: {xb.shape}, Target shape: {yb.shape}")
print(f"Input batch:\n{xb} \n\nTarget batch:\n{yb}")


from torch import nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    """Simplest possible language model: bigram model."""

    def __init__(self, vocab_size):
        super().__init__()

        # create a token embedding table, with vocab_size rows and columns
        # creates a thin wrapper of a tensor with vocab_size rows and vocab_size columns
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets = None):

        # index and targets are both (B, T) shaped tensor of integers
        # where B is the batch size and T is time i.e. the sequence length
        # B is 4, T is 8
        logits = self.token_embedding_table(index) # the output is (B, T, C) where C is "channel" i.e. vocab size
        # C = 65
        # it does this by looking up index in the embedding table, i.e. 43 will get the 42nd row of the table

        # logits is essentially the scores for the next token in the sequence
        # we are predicting what comes next based on a single identity of the current token i.e. tokens are not dependent on each other


        loss = None
        if targets is not None:
            # lets also calculate the loss function, which in this case will be cross entropy
            # cross entropy is a measure of difference between two probablity distributions for a given random variable or set of events
            # in this case, the probability distribution is the predicted logits, and the target is the actual next token
            # pytorch expets in form B x C x T
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        """Generate new tokens from the model resulting in (B, T+max_new_tokens)."""
        # index is a (B, T) tensor of integers in context
        for _ in range(max_new_tokens):
            # get logits (predictions) for the next token
            logits, loss = self(index)
            # get last time step 
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # softmax normalizes the logits to be between 0 and 1, and sum to 1

            # sample from the distribution
            # does a multinomial sampling i.e. binomial
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append to the context
            index = torch.cat([index, next_token], dim=1) # (B, T+1)

        return index

    
vocab_size = len(chars)
m = BigramLanguageModel(vocab_size)
out, loss = m(xb, yb)
print(out.shape)
print(f"Loss: {loss}")

# we create the first input, which will always be our start token
# equal to \n
input_indexes = torch.zeros((1, 1), dtype=torch.long)
generated = m.generate(input_indexes, 100)
batch = generated[0].tolist()
decoded = decode_string(batch, int_to_str)

print(decoded)