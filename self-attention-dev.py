import torch
from torch.nn import functional as F

# example 1
# imagine instead of ignoring all previous tokens we're able to consider all tokens up to the current one
# initially a basic way to do this would be to take the average of all the token embeddings up to the current one
torch.manual_seed(1337)
B, T, C = 4, 8, 4 # batch size, sequence length, vocab size

# create a random tensor of shape (B, T, C)
logits = torch.randn(B, T, C)

print(f"Logits shape: {logits.shape}")

# using a simple method of averaging ebfore we bring in the matrices
# bow = bag of words, i.e. we're averaging the words
x_bow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        logits_prev = logits[b, :t+1] # (t, c)
        x_bow[b, t] = torch.mean(logits_prev, dim=0) # (c,)

print(f"logits before bow: {logits[0]}")

print(f"x_bow: {x_bow[0]}")
print(f"Notice moving average")

# example 2
torch.manual_seed(42)
a = torch.ones(3, 3)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

print(f"\na: {a}")
print(f"b: {b}")
print(f"c: {c}")

# using the same examlpe but where a is a triangular matrix
# this allows us to ignore the future tokens by setting them to 0
a = torch.tril(torch.ones(3, 3))
c = a @ b
print(f"\na: {a}")
print(f"b: {b}")
print(f"c: {c}")

a = a / torch.sum(a, 1, keepdim=True) # normalize the rows
# this calculates the average of the previous tokens
c = a @ b
print(f"\na: {a}")
print(f"b: {b}")
print(f"c: {c}")

# this allows us to create a weights matrix
weights = torch.tril(torch.ones(T, T))
weights = weights / torch.sum(weights, 1, keepdim=True)
print(f"\nweights: {weights}")
x_bow2 = weights @ logits
print(f"x_bow2: {x_bow2[0]}")

# version 3 using softmax

tril = torch.tril(torch.ones(T, T))
weights = torch.zeros((T, T)) # if this isn't zero, we essentially will get an affinity between different tokens
# this allows softmax to ignore the future tokens, while normalizing the previous tokens
weights = weights.masked_fill(tril ==0, float('-inf'))# replaces all zeros with -inf
weights = F.softmax(weights, dim=1) # softmax also normalizes, resulting in a average of the previous tokens
print(f"\nweights: {weights}")

x_bow3 = weights @ logits
print(f"x_bow3: {x_bow3[0]}")


from torch import nn

# version 4 attention
#I want to look at all tokens in the past but in data dependent manner
#  all tokens will emit two vectors, a key and a query, they query answers "what am i looking for"
# and key answers "what do I contain"
# such that when I complete the dot product between the last query and all the keys, i get my weights matrix


torch.manual_seed(1337)
B, T, C = 4, 8, 32 # batch size, sequence length, vocab size
x = torch.randn(B, T, C)

# we implement a single head of self attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, head_size)
q = query(x) # (B, T, head_size)

weights = q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T) such that for each row of B, we get T x T affinities

tril = torch.tril(torch.ones(T, T))
weights = weights.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(weights, dim=1)

v = value(x) # (B, T, head_size)
out = weights @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

print(f"weights: {weights[0]}")
print(f"out: {out[0]}")

# its worth noting, attention is just a communication mechanism, can be seen as a graph where by tokens are pointed to from themselves 
# and previous tokens
# furthermore,there's no indication of "space", "time" or "position", it simply acts over a set of vectors, hence why in our model
#  we must add positional embeddings
# batches are completely separate, and never interact with each other

# in other use cases such as sentiment analysis, we can delete the tril function, because we want all tokens to
# talk to one another to get analysis of the entire time
# this is called an "encoder" block wheraes with tril we have a "decoder" block e.g. an autoregressive model

# self-attention describes the example where k, q, v all come from x, but cross attention is where k. v come from a separate source of nodes

# finally from the self-attention paper, they imlpement a "scaled" attention whereby weights are divided by the square root of the head size
# this makes it so when input Q, K are unit variance, wei will also be of unit variance too, otherwise would simply softmax to the highest value
# this means variance would be in the order of head size, such that variance increases with head size
# i.e weights goes from q @ kT to q @ kT / sqrt(head_size)