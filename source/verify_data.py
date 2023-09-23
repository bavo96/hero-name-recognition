import glob

if __name__ == "__main__":
    hero_names = [
        f.strip() for f in open("./data/test_data/hero_names.txt", "r").readlines()
    ]
    print(f"Number of hero names in hero_names.txt: {len(hero_names)}")

    img_hero_files = glob.glob("./data/thumbnail/*.png")
    img_hero_names = [
        f.replace("./data/thumbnail/", "").replace(".png", "") for f in img_hero_files
    ]
    print(f"Number of hero names on the original page: {len(img_hero_names)}")

    diff = [f for f in hero_names if f not in img_hero_names]

    print(f"Names in hero_names.txt that are not on the original page: {len(diff)}")
