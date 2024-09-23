from pathlib import Path

import cv2
from fer import FER


class FacialSentimentDetection:
    def __init__(self):
        self.model = FER(mtcnn=False)
        self.image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".tiff",
            ".webp",
        ]
        self.video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"]

    def get_files_in_directory(self, directory):
        files = []
        directory_path = Path(directory)

        # Iterate over files in directory and subdirectories
        for file_path in directory_path.rglob("*"):
            file_extension = file_path.suffix.lower()
            if (
                file_extension in self.image_extensions
                or file_extension in self.video_extensions
            ):
                files.append(file_path)

        return files

    # Sentiment detetcion for an image file
    def image_sentiment_detection(self, file_path):
        image_file = cv2.imread(file_path)
        image_result = {}
        image_result["analysis"] = self.model.detect_emotions(image_file)
        image_result["dominant_emotion"], image_result["emotion_score"] = (
            self.model.top_emotion(image_file)
        )
        return str(image_result)

    # Sentiment detection for a video file
    def video_sentiment_detection(self, file_path):
        cap = cv2.VideoCapture(file_path)

        results = []
        # Analyze sentiment in each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            analysis = self.model.detect_emotions(frame)
            # Ignore frames if analysis is empty i.e., no faces detected in the frame
            if analysis:
                frame_result = {}
                frame_result["analysis"] = analysis
                frame_result["dominant_emotion"], frame_result["emotion_score"] = (
                    self.model.top_emotion(frame)
                )
                results.append(frame_result)

        cap.release()
        return str(results)

    def batch_sentiment_detection(self, file_paths):
        return [
            {"file_path": str(file_path), "result": self.sentiment_detection(file_path)}
            for file_path in file_paths
        ]

    def sentiment_detection(self, file_path):
        file = Path(file_path)
        if file.suffix.lower() in self.image_extensions:
            return self.image_sentiment_detection(file_path)
        elif file.suffix.lower() in self.video_extensions:
            return self.video_sentiment_detection(file_path)
        return None

    def sentiment_detection_directory(self, input_directory):
        return self.batch_sentiment_detection(
            self.get_files_in_directory(input_directory)
        )
