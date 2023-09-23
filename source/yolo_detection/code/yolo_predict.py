from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(
        "runs/detect/train/weights/best.pt"
    )  # load a pretrained model (recommended for training)

    # Use the model
    results = model.predict(
        # "test_data/test_images/Ahri_278220660753197_round6_Ahri_06-02-2021.mp4_10_2.jpg"
        "../../test_data/test_images/Darius_1115082439004174_round3_Darius_05-19-2021.mp4_10_1.jpg",
        save=True,
    )  # predict on an image

    print(results)
