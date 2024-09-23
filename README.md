## Facial Sentiment Detection

This project detects facial emotions from images and videos, identifying sentiments like happy, sad, angry, etc. The output includes both the detected emotions and their associated percentages, along with bounding box coordinates for faces.

### Installing requirements

1. Install pipenv

```
pip install pipenv
```

2. Activate the virtual environment

```
pipenv shell
```

3. Install dependencies

```
pipenv install
```

### Starting the server

You can start the backend server by running the following command. This will set up a REST API that allows clients to send images or videos for facial sentiment analysis.

```
python -m facial_sentiment_detection.backend.server
```

### Client example

A simple Python client is included to demonstrate how to interact with the backend server.

To run the client example, execute the following command:

```
python client_example.py
```

### Command line tool

The command-line interface (CLI) allows you to run sentiment detection on individual files or entire directories without needing to use the server. The ```--output_directory``` argument is required in both cases to specify where the result files will be saved.

#### Single File Sentiment Detection

To perform sentiment detection on a single file, use the ```--input_file``` argument followed by the path to the image or video file

```
python cmd_line_interface_example.py --output_directory "outputs" --input_file "inputs/test_video.mp4"
```

#### Directory Sentiment Detection

To run sentiment detection on all files in a directory, use the ```--input_directory``` argument followed by the path to the directory containing the input files

```
python cmd_line_interface_example.py --output_directory "outputs" --input_directory "inputs"
```
