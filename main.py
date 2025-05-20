import torch
from models import BigramModel, SPECIAL_CHAR


model = BigramModel("./names.txt")
model.visualize(".")
for i in range(10):
    word = ""
    while len(word) == 0 or word[-1] != SPECIAL_CHAR:
        word += model.predict(word)
    print(word[:-1])
