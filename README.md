# HERO NAME RECOGNITION

## Installation
Prerequisites
 + python 3.8.10
 + torch 1.13.0
 + [lightglue](https://github.com/cvg/LightGlue) (for solution 1, 3)
 + [yolov8](https://github.com/ultralytics/ultralytics) (for solution 2, 3)
 + [open-metric-learning](https://github.com/OML-Team/open-metric-learning) (for solution 2)
 + GPU card

To install necessary packages, run the following command line
```
chmod +x ./install.sh
./install.sh
```
## Usage
- Crawl hero thumbnails: 
```python
cd hero-name-recognition/source/ && python3 crawl_data.py
```
- Train yolov8: 
```python
cd hero-name-recognition/source/yolo_detection/code && python3 yolo_train.py
```
- Run solution 1, 2, 3 on `test_data/test_images/`: 
```python
cd hero-name-recognition/source/ && python3 solution1.py
cd hero-name-recognition/source/ && python3 solution2.py
cd hero-name-recognition/source/ && python3 solution3.py
```
- Run solution 3 on custom `test_images/`:
```python
cd hero-name-recognition/source/ && python3 main.py <path-to-test-images-folder> <path-to-output.txt>

```
