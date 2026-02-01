"""
Copyright (c) 2019, Fabian Heinemann, Gerald Birk, Birgit Stierstorfer;
Boehringer Ingelheim Pharma GmbH & Co KG
All rights reserved.
"""

import os
import time
import argparse
import yaml
import pandas as pd
from cnn_utils import cnn_utils


def main(args):
    # Open the configuration YAML file
    with open(args.config, "r") as file:
        start = time.time()
        config_yaml = yaml.safe_load(file)

        # Only Fibrosis
        score = "Fibrosis"

        # Classes
        list_of_classes = ["0", "1", "2", "3", "4", "ignore"]

        # Score name for output
        score_name = "Fibrosis_score"

        # Model
        model_file = config_yaml["models"]["fibrosis_model"]

        # Thresholds
        thresholds_file = config_yaml["thresholds"]["fibrosis_thresholds_json"]

        # Tiles
        tile_path = config_yaml["tiles"]["fibrosis_tile_path"]

        # Results
        results_path = config_yaml["results"]["results_path"]
        experiment_name = config_yaml["results"]["experiment_name"]

        # Print arguments
        print("\n-----------------------------\n")
        print(f"score:\t\t{score}\n")
        print(f"model:\t\t{os.path.basename(model_file)}\n")
        print(f"thresholds_file:\t{os.path.basename(thresholds_file)}\n")
        print(f"results_path:\t{results_path}\n")
        print(f"experiment_name:\t{experiment_name}\n")
        print("-----------------------------")

        print(f"\nCurrent score: {score}")

        # Create cnn_utils object
        cnn_utils_obj = cnn_utils(
            model_path="",
            model_file_name=model_file,
            tile_path=tile_path,
            results_path=results_path,
            list_of_classes=list_of_classes
        )

        # Initialize model
        print("\nInitializing CNN...")
        cnn_utils_obj.initialize_model(load_pretrained_model=True)
        print("Model loaded.\n")

        # Classify tiles
        classification_result = cnn_utils_obj.classify_tiles()

        # Process results
        classification_result = cnn_utils_obj.process_results(classification_result)

        # Save detailed results
        detailed_file = (
            cnn_utils_obj.results_path
            + experiment_name
            + "_"
            + score_name
            + ".csv"
        )
        classification_result.to_csv(
            detailed_file,
            index=False,
            sep=";",
            decimal=".",
            float_format="%.2f"
        )
        print(f"Details saved to: {detailed_file}")

        # Generate summary results
        summary_result = cnn_utils_obj.generate_summary_results(
            classification_result,
            score_name,
            thresholds_file
        )

        summary_result.drop(
            columns=["n_tiles", "average_uncertainty"],
            inplace=True
        )

        # Save summary
        summary_file = results_path + experiment_name + "_summary.csv"
        summary_result.to_csv(
            summary_file,
            index=False,
            sep=";",
            decimal=".",
            float_format="%.2f"
        )
        print(f"Summary saved to: {summary_file}")

        # Print elapsed time
        end = time.time()
        print(f"Time elapsed: {end - start:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        action="store",
        dest="config",
        help="Filename of config file (*.yaml)",
        required=True
    )
    args = parser.parse_args()
    main(args)
