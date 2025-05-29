import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from evalutils.exceptions import ValidationError
import random
from typing import Dict
import json

execute_in_docker = False

class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print(path)
        if not path.is_file():
            raise IOError(
                f"Could not load {fname} using {self.__class__.__qualname__}."
            )
        return [{"path": fname}]

    def hash_video(self, input_video):
        pass

class UniqueVideoValidator(DataFrameValidator):
    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )

class SurgVU_classify(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-understanding-answer.json") if execute_in_docker else Path("./output/surgical-understanding-answer.json"),
            validators=dict(
                input_video=(
                    UniquePathIndicesValidator(),
                )
            ),
        )

        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights                         ###
        ###                                                                                                     ###

    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case
        # Detect and score candidates
        scored_candidates = self.predict(case.path)

        # Write resulting candidates to result.json for this case
        return scored_candidates

    def save(self):
        print('Saving prediction results to ' + str(self._output_file))
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def predict(self, fname: str) -> Dict:
        """
        Inputs:
        fname -> video file path

        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation
        """

        # Define the path to the question JSON file
        question_file = "./test/question.json"

        # Load the question from the JSON file
        with open(question_file, 'r') as f:
            question_data = json.load(f)

        print('Video file to be loaded: ' + str(fname))
        print('Question loaded: ' + str(question_data))

        # Generate fixed JSON output incorporating the question
        fixed_output = {
            "answer": "Esophageal hiatus hernia with gastric fundus entrapment.",
            "question": question_data.get("question")
        }
        return fixed_output

if __name__ == "__main__":
    SurgVU_classify().process()
