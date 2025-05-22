import torch
import matplotlib.pyplot as plt
from models import BigramModel, SimpleNetwork, SPECIAL_CHAR, BatchedNetwork


model = BatchedNetwork(fill=3)
model.load_data("./names.txt")
losses = model.train(itterations=100_000, batch_size=32)
plt.plot(losses)
plt.show()
model.visualize(".")
for i in range(10):
    word = ""
    while len(word) == 0 or word[-1] != SPECIAL_CHAR:
        word += model.predict(word)
    print(word[:-1])
