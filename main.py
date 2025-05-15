"""This file is to run the validation of the super-resolution performed in the GenAI Data."""

import os
import json
import subprocess
from pathlib import Path
import polars as pl
import logging
from typing import Literal

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


VALID_CALCULATIONS = [
    "scale-invariant-feature-transform",
    "multi-scale-structural-similarity-index",
    "peak-signal-to-noise-ratio",
    "learned-perceptual-image-patch-similarity",
]


def get_files_from_data(
    original_source: str,
    super_resolution_source: str,
) -> dict[str, tuple[str, str]]:
    """
    Get original and super-resolution image pairs.

    Args:
        source: The source of the data (e.g., "GenAI", "Control")
        multiplier: The super-resolution multiplier (e.g., "2x", "4x")

    Returns:
        Dictionary mapping file keys to tuples of (original_image_path, super_resolution_image_path)
    """
    return_dict = {}  # FileKey -> (original_image_path, super_resolution_image_path)

    # List all files in original and super-resolution directories
    sr_files = sorted(list(Path(original_source).glob("*")))
    sr_multiplier_files = sorted(list(Path(super_resolution_source).glob("*")))

    # Validate file count
    assert len(sr_files) == len(sr_multiplier_files), (
        f"The number of files in the `{original_source}` and `{super_resolution_source}` directories are not the same"
    )

    # Match original and super-resolution files
    for sr_file, sr_multiplier_file in zip(sr_files, sr_multiplier_files):
        return_dict[sr_file.stem] = (sr_file.resolve(), sr_multiplier_file.resolve())

    return return_dict


def validate_single_file(
    image_key: str,
    original_image_path: str,
    super_resolution_image_path: str,
    calculation_type: str,
    multiplier: str,
) -> dict:
    """For each file, we must first move into the calculation type's directory.
    The directory is found under `validations/{calculation_type}/main.py`.

    There, we will run the command "uv run python main {original_image_path} {super_resolution_image_path} {output_file_path}"

    The output file path will contain a JSON with the results.
    We want to return the output JSON value as a dictionary.
    """

    # Folder to run the validation in
    validation_folder = f"validations/{calculation_type}"

    logger.info(f"Running validation in {validation_folder}")

    # First we move into the calculation type's directory
    os.chdir(validation_folder)

    # Determine an output file path, pased on the image key
    output_file_path = f".output/{image_key}__{multiplier}.json"
    subprocess.call(["mkdir", "-p", ".output"])

    # Then we run the command
    output = subprocess.call([
        "uv",
        "run",
        "python",
        "main.py",
        original_image_path,
        super_resolution_image_path,
        "-o",
        output_file_path,
    ])

    logger.info(f"Output: {output}")

    # Then we read the output file
    with open(output_file_path, "r") as f:
        final_json = json.load(f)

    # Return to the original directory
    os.chdir("../..")

    return final_json


def create_dataset_for_multiplier(
    all_files_dict: dict[str, tuple[str, str]], multiplier: str, image_source: str
) -> pl.DataFrame:
    """
    Iterate over the data and run the validation for each file.
    This will also attach some metadata to the results.

    Finally, we create a dataframe with the results.
    """
    all_results = []
    for image_key, (
        original_image_path,
        super_resolution_image_path,
    ) in all_files_dict.items():
        image_dict = {
            "FileKey": image_key,
            "ImageSource": image_source,
            "OriginalImagePath": str(original_image_path),
            "SuperResolutionImagePath": str(super_resolution_image_path),
            "Multiplier": multiplier,
        }

        for calculation_type in VALID_CALCULATIONS:
            results = validate_single_file(
                image_key,
                original_image_path,
                super_resolution_image_path,
                calculation_type,
                multiplier,
            )
            for key, value in results.items():
                image_dict[f"{calculation_type}_{key}"] = value

        all_results.append(image_dict)

    # Then we convert the dictionary to a pandas dataframe
    dataset_multiplier_df = pl.DataFrame(all_results)

    return dataset_multiplier_df


def run_validation():
    # First we get all the files
    all_files_2x = get_files_from_data(
        original_source="./data/SR",
        super_resolution_source="./data/SR_2x",
    )

    # Create the dataset for the 2x multiplier
    dataset_2x_df = create_dataset_for_multiplier(
        all_files_dict=all_files_2x,
        multiplier="2x",
        image_source="GenAI",
    )

    # Then we save the dataframe to a CSV file
    dataset_2x_df.write_csv("analysis/2x_results.csv")

    # Then we get all the 4x files
    all_files_4x = get_files_from_data(
        original_source="./data/SR",
        super_resolution_source="./data/SR_4x",
    )

    # Create the dataset for the 4x multiplier
    dataset_4x_df = create_dataset_for_multiplier(
        all_files_dict=all_files_4x,
        multiplier="4x",
        image_source="GenAI",
    )

    # Then we save the dataframe to a CSV file
    dataset_4x_df.write_csv("analysis/4x_results.csv")

    ###### CONTROL DATASET ######
    all_files_control_2x = get_files_from_data(
        original_source="./data/DIV2K_valid_LR_bicubic/X2",
        super_resolution_source="./data/DIV2K_valid_HR",
    )

    # Create the dataset for the control
    dataset_control_2x_df = create_dataset_for_multiplier(
        all_files_dict=all_files_control_2x,
        multiplier="2x",
        image_source="Control",
    )

    # Then we save the dataframe to a CSV file
    dataset_control_2x_df.write_csv("analysis/control_results_2x.csv")

    # 4x control dataset
    all_files_control_4x = get_files_from_data(
        original_source="./data/DIV2K_valid_LR_bicubic_X4/X4",
        super_resolution_source="./data/DIV2K_valid_HR",
    )

    # Create the dataset for the control
    dataset_control_4x_df = create_dataset_for_multiplier(
        all_files_dict=all_files_control_4x,
        multiplier="4x",
        image_source="Control",
    )

    # Then we save the dataframe to a CSV file
    dataset_control_4x_df.write_csv("analysis/control_results_4x.csv")


if __name__ == "__main__":
    run_validation()
