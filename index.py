from flask import Flask, request
from flask.json import jsonify
from detect import faceDetect


# Flask setup

app = Flask(__name__)


@app.route("/")
def check():
    return jsonify({"message": "Image process ready"})


@app.route("/", methods=["POST"])
def imageProcessPost():
    try:
        input_json = request.get_json(force=True)
        url = input_json["url"]

        imgResult = faceDetect(url)

        return jsonify(
            {"data": imgResult, "status": 200, "message": "success detecting",}
        )

    except Exception as e:
        return jsonify({"message": "failed detecting", "status": 500}), 500


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")

