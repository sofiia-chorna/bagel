import ast
import pickle


class Concept_Manager:
    def __init__(self):
        pass

    def load(self, path: str):
        with open(path, "rb") as f:
            df = pickle.load(f)

        df["concepts"] = df["concepts"].apply(ast.literal_eval)

        return df

    def ask_llm(self, params: dict):
        pass
