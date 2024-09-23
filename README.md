## Facial Sentiment Detection

### Installing requirements

```
pip install requirements.txt
```

### Starting the server

```
python -m facial_sentiment_detection.backend.server
```

### Client example

```
python client_example.py
```

### Command line tool

Sentiment detection for a single file
```
python cmd_line_interface_example.py --output_directory "outputs" --input_file "inputs/test_video.mp4"
```

Sentiment detection for all files in a directory
```
python cmd_line_interface_example.py --output_directory "outputs" --input_directory "inputs"
```
