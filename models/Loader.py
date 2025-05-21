import os
from typing import List


class Loader:
    @staticmethod
    def load_from_file(path: str) -> List[str]:
        if not os.path.exists(path):
            raise ValueError(f"File path does not exists: {path}")
        file_content = []
        with open(path, "r") as f:
            for line in f:
                file_content.append(line[:-1])  # ignore newline or eof chars
        print(file_content[:10])
        return file_content
