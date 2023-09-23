import re
import shutil
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

if __name__ == "__main__":
    # Create thumbnail folder
    Path("./data/thumbnail/").mkdir(parents=True, exist_ok=True)

    # Access leagueoflegends page and crawl hero thumbnails
    html_page = requests.get(
        "https://leagueoflegends.fandom.com/wiki/Champion_(Wild_Rift)"
    ).content
    soup = BeautifulSoup(html_page, "html.parser")
    thumbnail_links = []
    columntemplate = soup.find("div", attrs={"class": re.compile("columntemplate")})
    for span in tqdm(
        columntemplate.findAll(
            "span",
            attrs={"class": "inline-image label-after champion-icon"},
        )
    ):
        thumb_link = (
            span.find("span", attrs={"class": "border"})
            .find("a")
            .find("img")["data-src"]
            .replace("/20?", "/400?")
        )
        local_img_name = [f for f in thumb_link.split("/") if ".png" in f][0]
        local_img_name = local_img_name.replace("_OriginalSquare_WR.png", "").replace(
            "%27", ""
        )
        print(f"Crawling {local_img_name} thumbnail...")
        response = requests.get(thumb_link, stream=True)
        with open(f"./thumbnail/{local_img_name}.png", "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
