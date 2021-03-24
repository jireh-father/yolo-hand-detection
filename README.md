# YOLO-Hand-Detection
Scene hand detection for real world images.

![Hand Detection Example](readme/export.jpg)

## Install
```bash
git clone https://github.com/jireh-father/yolo-hand-detection.git
cd yolo-hand-detection
pip install -r requirements.txt
```

## Download models
./models/download-models.sh

## Running flask
```bash
python web.py

```

## Test
1. open your browser

2. connect http://localhost:8888/hand_detection?image_path=images/cathryn-lavery-fMD_Cru6OTk-unsplash.jpg

3. result

```json
{
  "nums_of_hand": 1, 
  "output_path": "./output_images/30474eac-c5ec-4c33-9d34-69e86781ff3f.jpg"
}
```
4. check result file in output_images folder

## Options
```python
# 11 line in web.py

size = 416 # 입력 이미지 크기를 작게 할수록 빠르고 대신 정확도는 낮아짐.
confidence = 0.25 # confidence 를 낮출수록 손을 많이 찾아내지만 오탐이 많아짐. 0~1값.
```

```python
# 10 line in web.py

# 네트워크를 바꿔가면서 속도와 성능이 맘에 드는걸 찾아보세요.
network = 'normal' # 
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

```