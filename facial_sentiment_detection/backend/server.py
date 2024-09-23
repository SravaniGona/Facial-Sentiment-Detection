from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import CustomInput, ImageResult, ResponseModel

from ..ml.model import FacialSentimentDetection

model = FacialSentimentDetection()
server = MLServer(__name__)


@server.route("/sentimentdetection", input_type=DataTypes.CUSTOM)
def sentiment_detection(inputs: list[CustomInput], parameters):
    files = [f.input for i, f in enumerate(inputs)]
    results = model.batch_sentiment_detection(files)
    image_results = [
        ImageResult(file_path=res["file_path"], result=res["result"]) for res in results
    ]
    response = ResponseModel(results=image_results)
    return response.get_response()


server.run()
