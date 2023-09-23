# SOLUTION 3: yolo detection + feature matching

import glob
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms
from lightglue import DISK, LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from ultralytics import YOLO

# # SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features="superpoint").eval().cuda()  # load the matcher


def feature_matching(template_path, target_image_path):
    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = load_image(template_path, resize=50).cuda()
    image1 = load_image(target_image_path).cuda()

    # extract local features
    feats0 = extractor.extract(
        image0
    )  # auto-resize the image, disable with resize=None
    feats1 = extractor.extract(image1)

    # match the features
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = (
        feats0["keypoints"],
        feats1["keypoints"],
        matches01["matches"],
    )
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # Unblock below code to get the original keypoints and the visualization code
    # hero_name = template_path.replace("./data/thumbnail/", "").replace(".png", "")
    # axes = viz2d.plot_images([image0, image1])
    # viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    # viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
    # plt.savefig(f"./result/{hero_name}_matches.jpg")
    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(
    #     matches01["prune1"]
    # )
    # viz2d.plot_images([image0, image1])
    # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
    # plt.savefig(f"./result/{hero_name}_keypoints.jpg")

    return len(m_kpts0)


def inference_feature_matching(image_path, hero_thumbnails):
    max_points = {}
    for thumbnail in tqdm(hero_thumbnails):
        hero_name = thumbnail.replace("./data/thumbnail/", "").replace(".png", "")
        match_points = feature_matching(thumbnail, image_path)
        max_points[hero_name] = match_points
    max_points = dict(sorted(max_points.items(), key=lambda item: item[1]))
    max_key = max(max_points, key=max_points.get)
    return max_key, max_points[max_key]


if __name__ == "__main__":
    # Path("./result/").mkdir(parents=True, exist_ok=True)
    hero_thumbnails = glob.glob("./data/thumbnail/*png")  # Reference data
    test_data = open("./data/test_data/test.txt", "r").readlines()  # Target data
    true_prediction = 0
    total_samples = len(test_data)
    accuracy = 0
    # Load a pretrained YOLOv8n model
    model = YOLO("./yolo_detection/code/runs/detect/train/weights/best.pt")

    for sample in tqdm(test_data):
        filename, label = sample.split()

        results = model.predict(
            f"./data/test_data/test_images/{filename}",
            # save=True,
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
            hero_img_path = f"./hero_temp_img.jpg"
            cv2.imwrite(hero_img_path, hero_img)

            # Match target box with hero's thumbnails
            predicted_name, match_keypoints = inference_feature_matching(
                hero_img_path, hero_thumbnails
            )

            print(f"Target image: {filename}. Hero's name: {label}")
            print(f"Prediction: {predicted_name}. Keypoints: {match_keypoints}")

            if predicted_name == label:
                true_prediction += 1

    print(f"True prediction: {true_prediction}")
    print(f"Total sample: {total_samples}")
    print(f"Accuracy: {true_prediction}/{total_samples}")
