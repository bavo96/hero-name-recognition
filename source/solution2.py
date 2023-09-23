# SOLUTION 2: yolo detection + image similarity

import glob

import cv2
import torch
import torchvision.transforms as transforms
from oml.models import ViTUnicomExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from ultralytics import YOLO


def get_sim_features(img_path, model):
    img = im_reader(img_path)  # put path to your image here
    img_tensor = [transforms(img)]
    batch_tensor = torch.stack(img_tensor, dim=0).to(device)
    features = model(batch_tensor).detach()
    return features


if __name__ == "__main__":
    hero_thumbnails = glob.glob("./data/thumbnail/*png")  # Reference data
    test_data = open("./data/test_data/test.txt", "r").readlines()  # Target data
    true_prediction = 0
    total_samples = len(test_data)
    accuracy = 0

    # Load a pretrained YOLOv8n model
    yolo_model = YOLO("./yolo_detection/code/runs/detect/train2/weights/best.pt")

    # Load image similarity model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sim_model = (
        ViTUnicomExtractor.from_pretrained("vitl14_336px_unicom").eval().to(device)
    )

    transforms, im_reader = get_transforms_for_pretrained("vitl14_336px_unicom")

    for sample in tqdm(test_data):
        filename, label = sample.split()
        max_similarity = 0
        target_hero = ""

        results = yolo_model.predict(
            f"./data/test_data/test_images/{filename}",
            conf=0.25,
            verbose=False,
        )  # predict on an image
        for r in results:
            # Choose box with lowest x coordinate
            data = r.boxes.data
            x_ele = data[:, 0].reshape(1, -1)
            min_elements, min_idx = torch.min(x_ele, dim=1)
            target_box = data[min_idx.cpu()][0]

            # Crop target box with target hero
            bb, cls, conf = target_box[:4], target_box[4], target_box[5]
            bb = [int(coor) for coor in bb]
            img = cv2.imread(f"./data/test_data/test_images/{filename}")
            hero_img = img[bb[1] : bb[1] + bb[3], bb[0] : bb[0] + bb[2]]
            hero_img_path = f"./temp_hero_img.jpg"
            cv2.imwrite(hero_img_path, hero_img)
            hero_img_feat = get_sim_features(hero_img_path, sim_model).reshape(1, -1)

            for thumb_path in hero_thumbnails:
                thumb_img_feat = get_sim_features(thumb_path, sim_model).reshape(1, -1)
                similarity_score = cosine_similarity(
                    hero_img_feat.cpu(), thumb_img_feat.cpu()
                )[0][0]
                # print(
                #     f"Thumbnail path: {thumb_path}. Similarity score: {similarity_score:.2f}"
                # )
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    target_hero = thumb_path.split("/")[3].split(".")[0]

            print(f"Target image: {filename}. Hero's name: {label}")
            print(f"Prediction: {target_hero}. Similarity score: {max_similarity}")
            if target_hero == label:
                true_prediction += 1

    print(f"True prediction: {true_prediction}")
    print(f"Total sample: {total_samples}")
    print(f"Accuracy: {true_prediction}/{total_samples}")
