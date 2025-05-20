from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def visualize(self, path: str):
        """Visualize the model in some way and save it to the directory specified"""
        pass

    def predict(self, content: str) -> str:
        """Predict the next character in the string"""
        pass
