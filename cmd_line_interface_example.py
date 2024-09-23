import argparse
from pathlib import Path

from facial_sentiment_detection.ml.model import FacialSentimentDetection


def write_res_to_dir(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        with open(out_dir / (Path(result["file_path"]).stem + ".txt"), "w") as f:
            f.write(result["result"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, help="The path to the input file", default=None
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        help="The path to the directory containing input files",
        default=None,
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="The path to the directory to save the sentiment analysis",
        required=True,
    )
    args = parser.parse_args()

    input_file, input_directory, output_directory = (
        args.input_file,
        args.input_directory,
        args.output_directory,
    )
    if input_file is None and input_directory is None:
        raise ValueError("Either --input_file or --input_directory must be provided")

    if input_file and input_directory:
        raise ValueError(
            "Only one of --audio_file or --audio_directory can be provided"
        )

    model = FacialSentimentDetection()
    if input_file:
        results = [
            {"file_path": input_file, "result": model.sentiment_detection(input_file)}
        ]
    elif input_directory:
        results = model.sentiment_detection_directory(input_directory)

    write_res_to_dir(results, output_directory)


if __name__ == "__main__":
    main()
