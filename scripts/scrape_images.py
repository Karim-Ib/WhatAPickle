"""
Build a test dataset using Bing Image search.

Downloads into data/pickle/ and data/non_pickle/.
Run once, then MANUALLY review both folders — remove mislabeled images.
"""

import argparse
import shutil
from pathlib import Path
from bing_image_downloader import downloader


PICKLE_QUERIES = [
    "pickles on a plate",
    "jar of dill pickles",
    "pickle spear close up",
    "fried pickles appetizer",
    "burger with pickles",
    "gherkins in a bowl",
    "pickle meme funny",
    "homemade pickled cucumbers",
]

NON_PICKLE_QUERIES = [
    "dinner plate food photo",
    "burger food photo",
    "fresh salad bowl",
    "pasta dish",
    "sushi plate",
    "pizza slice",
    "fresh cucumber slices",
    "cake dessert",
    "selfie person",
    "landscape nature",
    "cute cat photo",
    "funny meme",
]

IMAGES_PER_QUERY = 15


def scrape_class(queries: list, label: str, data_dir: Path, per_query: int):
    label_dir = data_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    # bing_image_downloader saves into its own subfolder structure,
    # so we download to a temp dir then flatten into our label folder
    tmp_dir = data_dir / "_tmp_download"

    idx = len(list(label_dir.glob("*")))

    for query in queries:
        print(f"\n  Downloading: '{query}'")
        try:
            downloader.download(
                query,
                limit=per_query,
                output_dir=str(tmp_dir),
                adult_filter_off=False,
                force_replace=False,
                timeout=10,
            )
        except Exception as e:
            print(f"    Failed: {e}")
            continue

        # Flatten: bing downloader saves to tmp_dir/query_name/Image_N.jpg
        query_dir = tmp_dir / query
        if not query_dir.exists():
            continue

        for img_path in sorted(query_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
                ext = img_path.suffix.lower()
                dest = label_dir / f"{label}_{idx:04d}{ext}"
                shutil.move(str(img_path), str(dest))
                idx += 1

    # Clean up temp dir
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"\n  Total {label} images: {idx}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--per-query", type=int, default=IMAGES_PER_QUERY)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Downloading PICKLE images")
    print("=" * 60)
    scrape_class(PICKLE_QUERIES, "pickle", data_dir, args.per_query)

    print("\n" + "=" * 60)
    print("Downloading NON-PICKLE images")
    print("=" * 60)
    scrape_class(NON_PICKLE_QUERIES, "non_pickle", data_dir, args.per_query)

    print(f"\nDataset ready in {data_dir}/")
    print("IMPORTANT: Manually review both folders and remove mislabeled images!")


if __name__ == "__main__":
    main()