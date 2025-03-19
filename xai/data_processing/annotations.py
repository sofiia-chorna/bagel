from typing import Dict, List, Optional

from xai.utils.consts import IMAGENET_CLASS_TO_LABEL
from xai.utils.file import load_json, recursive_list_files
from xai.utils.logger import logger


def process_annotations(
    annotations_path: str, labels_to_keep: Optional[List[str]]
) -> List[Dict[str, int | str | List[str]]]:

    concepts = []
    index_counter = 0

    for annotation_file_path in recursive_list_files(annotations_path):
        logger.info(f"Processing {annotation_file_path} ...")
        data = load_json(annotation_file_path)
        label = IMAGENET_CLASS_TO_LABEL.get(data["imagenet_category_id"], -1)

        if labels_to_keep is None or str(label) in labels_to_keep:
            concept_entries = [
                {
                    "index": index_counter + i,
                    "label": int(label),
                    "concepts": img_entry["categories"],
                    "filename": img_entry["file_name"],
                }
                for i, img_entry in enumerate(data["images"])
            ]

            concepts.extend(concept_entries)
            index_counter += len(data["images"])

    return concepts


def filter_class_filepaths(labels_to_keep: Optional[List[str]]) -> Dict[str, str]:
    class_filepaths = load_json("xai/data_processing/imagenet_class_filepaths.json")
    if labels_to_keep:
        return {
            str(key): value
            for key, value in class_filepaths.items()
            if key in labels_to_keep
        }
    return class_filepaths
