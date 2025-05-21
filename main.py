import torch
from models import BigramModel, SimpleNetwork, SPECIAL_CHAR


model = SimpleNetwork(fill=3)
model.load_data("./names.txt")
model.train()
model.visualize(".")
for i in range(10):
    word = ""
    while len(word) == 0 or word[-1] != SPECIAL_CHAR:
        word += model.predict(word)
    print(word[:-1])
