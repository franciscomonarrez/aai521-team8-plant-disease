# AAI-521 Team 8: Plant disease classification

Final project for AAI-521 (Applied Computer Vision) at the University of San Diego.

We classify leaf images into 38 crop/disease classes. Models are trained on the color subset of PlantVillage ([Kaggle: `abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)), which is lab photos of single leaves on a plain background. [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) is used only as a second test set of field photos, because models that do well on PlantVillage usually drop a lot on real-world pictures. We report the two test sets separately.

PlantVillage is CC BY-NC-SA 4.0 on Kaggle, so the images (and anyone's Kaggle key) stay out of the repo.

## Team

| Person | Work |
| --- | --- |
| Francisco | Data pipeline, splits, PlantDoc cleanup, EDA (this branch) |
| Mayur | Models: baseline CNN, ResNet50, EfficientNet |
| Eric | Evaluation on the PlantVillage test split and PlantDoc |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The CPU-only PyTorch build is enough for the data code and the notebook:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Kaggle credentials

PlantVillage is downloaded with the Kaggle API. Either export `KAGGLE_USERNAME` and `KAGGLE_KEY`, or create a token on your [Kaggle settings](https://www.kaggle.com/settings) page (API, Create New Token), save it as `~/.kaggle/kaggle.json`, and run `chmod 600` on it. Newer Kaggle clients also accept `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`. Don't put the key in the repo.

To check that the credentials are found without downloading anything:

```bash
python -m src.data.download --dry-run
```

## Data

From the repo root:

```bash
python -m src.data.download
python -m src.data.splits
```

`download` puts everything under `data/raw/`, which is gitignored:

- `plantvillage/color/<class>/`: the color images only (the grayscale and segmented copies in the zip are skipped).
- `plantvillage-leaf-map.json`: leaf ids from the original PlantVillage repo ([spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)), pinned to one commit and checked by SHA-256.
- `plantdoc/train/` and `plantdoc/test/`: PlantDoc cloned at commit `5467f60` (2,578 images).

The leaf map and PlantDoc don't need a Kaggle key. Six PlantDoc files have the same name as another file except for letter case (for example `CAR1.jpg` and `car1.jpg`). macOS and Windows can't store both, so the second one is saved as `<name>__case2.jpg` on every OS. If you cloned PlantDoc before this change, `python -m src.data.download --plantdoc-only` fixes the copy in place (about 1 MB download).

`splits` writes the files below. They are committed, so normally there is no need to rebuild them; if you do, `git diff --stat data/` should come back empty.

| File | Contents |
| --- | --- |
| `data/splits/train.csv`, `val.csv`, `test.csv` | 38,014 / 8,144 / 8,147 PlantVillage images, columns `path,label,group,fold` |
| `data/splits/plantdoc_test.csv` | 2,269 PlantDoc images in 27 classes, columns `path,label,source_split` |
| `data/plantdoc_label_map.csv` | PlantDoc to PlantVillage class mapping |
| `data/splits/*_imagelevel.csv` | The old image-level split, kept for comparison |

`data/plantdoc_excluded.csv` is an input to `splits` (see below). `python -m src.data.splits --plantdoc-only` rebuilds just the PlantDoc files and doesn't need PlantVillage. The CSVs are sorted by path, so shuffle in the DataLoader.

### PlantVillage split

Most PlantVillage leaves were photographed several times (median 4 photos per leaf), so a random image-level split puts the same leaf in train and test. With our first split, about 77% of val and test images shared a leaf with a train image. The split is now done by leaf. Each image gets a `group`: its leaf id from `leaf-map.json` when there is one (78% of images), otherwise its original file name. `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` assigns the groups to folds; folds 0-2 are test, 3-5 are val and 6-19 are train. No leaf is in more than one split, and each class is between 68% and 72% train. The `fold` column can be used for cross-validation inside train.

Two names had to be matched by hand: Apple black rot is called "Apple_Frogeye Spot" in the leaf map, and the Target Spot files named `Com.G_TgS_FL ...` are listed as `Com.G_FL ...`.

Caveats:

- Seven classes have no leaf ids (the four corn classes, healthy grape, squash powdery mildew and tomato mosaic virus), and five tomato/strawberry classes are only partly covered. In those classes repeat photos of a leaf can still land in different splits, so their test scores are probably a bit optimistic.
- The CSVs were made with scikit-learn 1.9.1. Older versions shuffle groups differently and give a different (still leaf-grouped) split.

### PlantDoc cleanup

PlantDoc was scraped from the web and has some label problems, so 309 of its 2,578 images are left out of `plantdoc_test.csv`. Each one is listed in `data/plantdoc_excluded.csv` with a reason:

- `file_name` (233): the file name points to a different disease, crop or plant, or to a symptom in a healthy folder. For example, 42 of the 116 corn rust images are named as southern rust, stripe rust or tar spot, and 57 of the 192 corn leaf blight images as gray leaf spot, bacterial leaf streak and other diseases.
- `duplicate` (34): an extra copy of a photo that is kept elsewhere with the same label.
- `conflicting_duplicate` (42): the same photo appears under different labels (33 photos, 14 of them under two different crops). We kept one copy only when the file names made the right label clear (16 photos) and dropped the rest.

Duplicates were found with MD5 and a perceptual hash, and every candidate pair was checked by eye. File names come from the original web pages, so they are a hint and not a verified label. Images with uninformative names were not checked.

In the label map, six mappings are marked `approximate` because the PlantDoc folder is broader than the PlantVillage class: apple rust, pepper leaf spot, corn leaf blight, corn rust, mosaic virus and yellow virus. Tomato spider mites has no usable PlantDoc images (both are comparison charts), and ten PlantVillage classes have no PlantDoc folder at all, so 11 classes have no field-photo test.

## Using the data

```python
from src.data.dataset import load_plantdoc, load_splits

loaders = load_splits(batch_size=32, num_workers=4)
images, labels = next(iter(loaders["train"]))  # [B, 3, 224, 224]

plantdoc = load_plantdoc(batch_size=32)
```

Training images get a random resized crop, a horizontal flip, a small rotation and color jitter. Val, test and PlantDoc images are resized to 256 on the short side and center-cropped to 224. Everything is normalized with ImageNet statistics, and EXIF rotation is applied when an image is loaded. Class indexes follow the alphabetical order in `src/data/labels.py`, so PlantDoc labels line up with the training classes. PlantDoc's own `train/` folder is test data here too, so don't train on it.

Score `test.csv` and `plantdoc_test.csv` separately and report macro-F1 and per-class recall along with accuracy, since PlantVillage is very imbalanced (citrus greening has 5,507 images, healthy potato 152). The PlantVillage test score is still a lab number; the drop on PlantDoc is the main result. To see how much the old leak inflated scores, a model can be trained on `train_imagelevel.csv` and scored on `test_imagelevel.csv`.

## EDA

`notebooks/01_eda.ipynb` has the exploratory analysis and takes about 25 seconds to run. It covers class balance, image size and color, repeated leaves and duplicate images, PlantDoc class counts, and previews of the transforms. The figures are also saved in `figures/`.

## Layout

```text
data/
  plantdoc_label_map.csv
  plantdoc_excluded.csv
  splits/
  raw/                (gitignored)
figures/
notebooks/01_eda.ipynb
src/data/
  download.py         Kaggle download, leaf map, PlantDoc clone
  splits.py           splits and the PlantDoc test CSV
  dataset.py          Dataset, transforms, DataLoaders
  duplicates.py       MD5 and pHash helpers
  labels.py           class names and the PlantDoc mapping
```
