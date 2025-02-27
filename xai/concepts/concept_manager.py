import pickle
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoProcessor, BlipForQuestionAnswering, BlipProcessor

from xai.utils.consts import DEVICE
from xai.utils.file import load_json
from xai.utils.logger import logger


class Concept_Manager:
    PROMPT_TEMPLATE: Dict[str, str] = {
        "color": "Is there evidence of the color {concept} in this image?",
        "texture": "Does this image contain a {concept} texture?",
        "abstract": "Is the concept of {concept} associated with this object?",
        "physical_parts": "Does this object visibly have {concept}?",
        "scene": "Is this object in a {concept} setting?",
        "material": "Is this object made of {concept}?"
    }

    def __init__(self):
        self.all_concepts = load_json("xai/concepts/concepts_set.json")

    def load(self, path: str):
        logger.info(f"Start loading concepts from: {path}")
        with open(path, "rb") as f:
            df = pickle.load(f)

        logger.info("End loading concepts")

        return df

    def ask_llm(self, dataset: Dataset, batch_size: int = 64) -> pd.DataFrame:
        logger.info("Start annotating")

        logger.info("Start loading LLM")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        model.to(DEVICE)

        processor: BlipProcessor = AutoProcessor.from_pretrained(
            "Salesforce/blip-vqa-base"
        )
        logger.info("End loading LLM")

        ds_size: int = len(dataset)

        df_image_concept = {i: {"label": None, "concepts": []} for i in range(ds_size)}

        # flat array of concepts
        concepts: List[str] = [
            item for sublist in self.all_concepts.values() for item in sublist
        ]

        logger.info(f"Start processing {ds_size} images for {concepts}")

        for batch_start in tqdm(range(0, ds_size, batch_size)):
            batch_end = min(batch_start + batch_size, ds_size)
            batch_items = [dataset[i] for i in range(batch_start, batch_end)]
            images = [item["image"] for item in batch_items]
            labels = [item["label"] for item in batch_items]

            for concept in concepts:
                logger.info(f"Processing {concept}...")

                prompts = [self._get_prompt_for_concept(concept) for _ in images]
                answers = self._process_batch(model, processor, prompts, images)

                for index, answer in enumerate(answers):
                    image_index = batch_start + index
                    df_image_concept[image_index]["label"] = labels[index]
                    if answer == "yes":
                        df_image_concept[image_index]["concepts"].append(concept)

        logger.info(f"End processing images")

        # convert the dictionary to a dataframe
        results = pd.DataFrame(
            list(df_image_concept.items()), columns=["index", "data"]
        )
        results["label"] = results["data"].apply(lambda x: x["label"])
        results["concepts"] = results["data"].apply(lambda x: x["concepts"])
        results = results.drop(columns=["data"])

        logger.info("End annotating")

        return results

    def _get_prompt_for_concept(self, concept: str) -> str:
        for category, template in self.PROMPT_TEMPLATE.items():
            if concept in self.all_concepts.get(category, []):
                return template.format(concept=concept)

        return f"Does this object have {concept}?"

    def _process_batch(
        self,
        model: BlipForQuestionAnswering,
        processor: BlipProcessor,
        prompts: List[str],
        images: List[str],
    ) -> List[str]:
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {
            key: value.to(DEVICE, non_blocking=True) for key, value in inputs.items()
        }

        with autocast("cuda"):
            outputs = model.generate(**inputs, max_new_tokens=3)

        answers = [
            processor.decode(output.cpu(), skip_special_tokens=True).lower()
            for output in outputs
        ]

        return answers
    
    def get_concept_categories(self) -> List[str]:
        return list(self.all_concepts.keys())

# singleton
concept_manager = Concept_Manager()
