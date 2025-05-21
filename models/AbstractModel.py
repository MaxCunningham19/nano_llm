from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def visualize(self, path: str):
        """Visualize the model in some way and save it to the directory specified"""
        pass

    @abstractmethod
    def predict(self, content: str) -> str:
        """Predict the next character in the string"""
        pass

    @abstractmethod
    def load_data(self, data: str):
        """Pass the path to the data and load it into the model"""
        pass

    @abstractmethod
    def train(self, **kwargs):
        """Train the model on the previously passed data"""
        pass
