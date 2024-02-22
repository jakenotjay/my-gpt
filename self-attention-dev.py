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
