from flask import Flask, render_template, request, redirect, url_for, session, send_file, Response
import cv2
from yolo import YOLO
import json
import uuid
import os

app = Flask(__name__)

network = 'normal'
size = 416
confidence = 0.25
output_dir = './output_images'
os.makedirs(output_dir, exist_ok=True)

if network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(size)
yolo.confidence = float(confidence)


@app.route("/hand_detection")
def hand_detection():
    image_path = request.args.get("image_path")
    print(image_path)
    mat = cv2.imread(image_path)

    width, height, inference_time, results = yolo.inference(mat)

    print("%s seconds: %s classes found!" % (round(inference_time, 2), len(results)))

    if len(results) < 1:
        return json.dumps({"nums_of_hand": 0})

    for detection in results:
        id, name, confidence, x, y, w, h = detection

        # draw a bounding box rectangle and label on the image
        color = (255, 0, 255)
        cv2.rectangle(mat, (x, y), (x + w, y + h), color, 1)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, color, 1)

        print("%s with %s confidence" % (name, round(confidence, 2)))
    output_path = os.path.join(output_dir, str(uuid.uuid4()) + ".jpg")
    cv2.imwrite(output_path, mat)

    return json.dumps({"nums_of_hand": len(results), "output_path": output_path})


if __name__ == "__main__":
    app.run(debug=True, threaded=False, host='0.0.0.0', port='8888')
