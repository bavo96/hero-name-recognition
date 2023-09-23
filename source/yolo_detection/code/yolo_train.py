from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    model.train(
        data="yolo.yaml",
        epochs=20,
        # close_mosaic=0,
        single_cls=True,
        val=True,
    )  # train the model

    # Use the model
    metrics = model.val(
        data="yolo.yaml",
        split="val",  # , conf=0.001, iou=0.6, rect=True, save_hybrid=True
    )  # evaluate model performance on the validation set
    print(metrics.confusion_matrix)
