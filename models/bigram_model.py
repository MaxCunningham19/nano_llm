import os
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List
from .Loader import Loader
from .AbstractModel import AbstractModel
from .constants import SPECIAL_CHAR


class BigramModel(Loader, AbstractModel):
    def __init__(self, normalization: int = 1):
        self.normalization = normalization

    def load_data(self, data_path: str):
        self.data = self.load_from_file(data_path)

    def train(self, **kwargs):
        self.__setup_counts(self.data)

    def __setup_counts(self, file_content: List[str]):
        unique_chars = set("".join(file_content))
        unique_chars.add(SPECIAL_CHAR)
        self.unique_chars = sorted(unique_chars)
        self.counts = torch.tensor([[max(0, self.normalization) for _ in range(len(unique_chars))] for _ in range(len(unique_chars))])
        self.s_to_i = {s: i for i, s in enumerate(unique_chars)}
        self.i_to_s = {i: s for i, s in enumerate(unique_chars)}
        for line in file_content:
            for c1, c2 in zip(line, line[1:]):
                self.counts[self.s_to_i[c1], self.s_to_i[c2]] += 1
            self.counts[self.s_to_i[line[-1]], self.s_to_i[SPECIAL_CHAR]] += 1
        row_totals = self.counts.sum(dim=1, keepdim=True)
        self.probs = self.counts / row_totals

    def visualize(self, path):
        plt.imshow(self.probs, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Probability")  # Add color scale

        plt.title("2D Probability Heatmap")
        plt.xlabel("Second Char")
        plt.xticks(ticks=range(len(self.unique_chars)), labels=self.unique_chars)
        plt.yticks(ticks=range(len(self.unique_chars)), labels=self.unique_chars)
        plt.ylabel("First Char")
        plt.savefig(f"{path}/heatmap.png")

    def predict(self, content: str):
        if len(content) == 0:
            content = [SPECIAL_CHAR]
        ix = self.s_to_i[content[-1]]
        probs = self.probs[ix]
        sample = torch.multinomial(probs, num_samples=1).item()
        return self.i_to_s[sample]
