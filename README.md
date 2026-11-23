# AAI-521 Team 8 — Plant disease image classifier

Applied Computer Vision (University of San Diego), Team 8 final project.

The task is to classify leaf images into crop-and-disease classes. Training images come from the PlantVillage **color** subset ([Kaggle: `abdallahalidev/plantvillage-dataset`](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)): lab photos of single leaves on a plain background, 38 classes. Field photos come from [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) and are held out as a second test set. Models that score very high on PlantVillage usually drop on field photos, because the lab background, framing, and lighting are not what a phone picture looks like. Report the two test sets separately.

PlantVillage on Kaggle is listed as CC BY-NC-SA 4.0. Do not commit the images or anyone's Kaggle key.

## Team roles

| Person | Work |
| --- | --- |
| Francisco | Data pipeline, leaf-grouped splits, exploratory analysis (this branch) |
| Mayur | Models: a baseline CNN compared with ResNet50 and EfficientNet |
| Eric | Evaluation on the PlantVillage test split and on PlantDoc |

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CPU-only PyTorch wheel is enough for the data loader and the notebook:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Kaggle credentials

PlantVillage is downloaded with the Kaggle API. Pick one method. Do not commit the key, and do not paste it into the repo.

1. Environment variables:

   ```bash
   export KAGGLE_USERNAME="your_kaggle_username"
   export KAGGLE_KEY="your_kaggle_key"
   ```

2. Token file: on [Kaggle settings](https://www.kaggle.com/settings) choose **API → Create New Token**, then:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

The current Kaggle client also accepts `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`. Either style is enough. The key stays outside the repo.

Check that the client can see the credentials without downloading:

```bash
python -m src.data.download --dry-run
```

## Download and splits

From the repository root:

```bash
python -m src.data.download
python -m src.data.splits
```

`download` writes:

- `data/raw/plantvillage/color/<class>/*.jpg` — RGB PlantVillage only. Grayscale and segmented copies in the Kaggle zip are not extracted.
- `data/raw/plantdoc/train/` and `data/raw/plantdoc/test/` — PlantDoc class folders, cloned from GitHub. No Kaggle key is required for this part.
- `data/raw/plantvillage-leaf-map.json` — the leaf IDs published with the original PlantVillage release ([`spMohanty/PlantVillage-Dataset`](https://github.com/spMohanty/PlantVillage-Dataset), pinned to commit `7f7ecc7`, checked by SHA-256). No Kaggle key is required for this part either. `splits` needs it.

`data/raw/` is gitignored.

`splits` writes:

| File | Contents |
| --- | --- |
| `data/splits/train.csv` | 38,014 PlantVillage images (70.0%), columns `path,label,group,fold` |
| `data/splits/val.csv` | 8,144 (15.0%) |
| `data/splits/test.csv` | 8,147 (15.0%) |
| `data/splits/plantdoc_test.csv` | PlantDoc images whose folder maps to a PlantVillage label, columns `path,label` |
| `data/plantdoc_label_map.csv` | Crosswalk, including classes that do not map |
| `data/splits/{train,val,test}_imagelevel.csv` | The earlier image-level split (38,013 / 8,146 / 8,146, `path,label`). It leaks leaves; kept only for comparison. Rebuild with `python -m src.data.splits --image-level`. |

**The PlantVillage split is by leaf, not by image.** PlantVillage photographed most leaves several times, so the earlier image-level split put the same leaf in train and in test: about 77% of val and test images shared a leaf with a train image. Now every image has a `group`:

- `leaf:<class>:<id>` when the original release's `leaf-map.json` knows which leaf the photo shows. That covers 42,511 images (78.3%) on 7,945 leaves.
- `file:<class>:<original file name>` otherwise (11,794 images). Byte-identical copies of one file under different Kaggle prefixes share a name, so they stay together too.

`StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` deals the 19,724 groups into 20 folds with similar class balance. Folds 0–2 are test, 3–5 are val, and 6–19 are train. No leaf and no group appears in two splits. Every class is in every split, with 68.4–71.5% of each class in train (smallest class: healthy potato, 104 / 24 / 24). The `fold` column is kept, so you can cross-validate on `train.csv` by holding out one or more of folds 6–19 at a time; those folds are leaf-grouped too.

The committed CSVs were generated with scikit-learn 1.9.1. Older scikit-learn releases shuffle groups inside `StratifiedGroupKFold` differently, so a rebuild with one of them can deal the groups into different folds. That split is still leaf-grouped, but it is not the committed one. Treat the committed CSVs as the reference, and run `git diff data/splits` after any rebuild. `splits` prints the scikit-learn version it used.

Two naming fixes are needed to match `leaf-map.json` to the Kaggle file names: Apple black rot is listed there as "Apple_Frogeye Spot", and Target Spot frames named `Com.G_TgS_FL 1234` are listed as `Com.G_FL 1234` (all 1,404 match). Batch prefixes such as `OCT_769` (citrus) are kept as part of the leaf ID.

Known gap: seven classes have no leaf metadata at all, so they can only be grouped by file: the four corn classes, healthy grape, squash powdery mildew, and tomato mosaic virus (6,483 images). Tomato yellow leaf curl, late blight, Septoria, healthy tomato, and strawberry leaf scorch are only partly covered (5,311 more images). Repeat shots of one leaf can still cross splits in those images, so treat per-class test scores for them as optimistic.

CSVs are sorted by path; shuffle during training in the DataLoader, not by reordering the file. PlantDoc's own train folder is still test data here. Do not train on it.

Ten PlantVillage classes have no PlantDoc counterpart (citrus greening, apple black rot, tomato target spot, and others). They are `status=unmapped` in `data/plantdoc_label_map.csv` and are absent from `plantdoc_test.csv`. Notes in that file explain the close calls, such as PlantDoc "Soyabean leaf" → `Soybean___healthy`, and strawberry leaf → healthy rather than leaf scorch.

Rebuild the label map without scanning images:

```bash
python -m src.data.splits --map-only
```

## Modeling and evaluation

```python
from src.data.dataset import load_plantdoc, load_splits

loaders = load_splits(batch_size=32, num_workers=4)
images, labels = next(iter(loaders["train"]))  # images: [B, 3, 224, 224]

plantdoc = load_plantdoc(batch_size=32)
```

Train augmentations are a random resized crop to 224, horizontal flip, a small rotation, and color jitter. Val, test, and PlantDoc use resize-shorter-side 256, center crop 224, and ImageNet normalization. Class indexes are alphabetical over the 38 PlantVillage names (`src.data.labels.CLASS_TO_IDX`), so a PlantDoc label lines up with the same index used in training.

For evaluation, score `data/splits/test.csv` and `data/splits/plantdoc_test.csv` separately. Accuracy on PlantVillage will be dominated by the largest classes (citrus greening, soybean healthy, tomato yellow leaf curl). Macro-F1 and per-class recall are the numbers that show whether the small classes, such as healthy potato, are actually learned. The in-dataset test split is leaf-grouped, so a model cannot score by recognizing a leaf it saw in training (except in the classes listed under "Known gap" above). It is still a lab number: same backgrounds, lighting, and framing as train. The field-photo drop is the result that matters. To show how much the old leak inflated scores, a model can also be scored on `test_imagelevel.csv`, but only if it was trained on `train_imagelevel.csv`.

Exploratory charts are in `notebooks/01_eda.ipynb` (outputs saved). On the standard color set that notebook measured 54,305 PlantVillage images, all 256×256, across 38 classes. The largest class (citrus greening) has 5,507 images and the smallest (healthy potato) has 152, an imbalance of about 36×. The leaf-grouped split is 38,014 / 8,144 / 8,147. The notebook's saved outputs and its split notes still describe the earlier image-level split (38,013 / 8,146 / 8,146) until it is re-run. PlantDoc contributes 2,578 mapped field photos across 28 of those classes; the other 10 classes have no field photos. PlantDoc widths range from about 100 px to 6,000 px, so the lab test score and the field test score are not the same experiment.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── plantdoc_label_map.csv      # tracked crosswalk
│   ├── splits/                     # tracked split CSVs
│   └── raw/                        # gitignored downloads
│       ├── plantvillage/color/
│       ├── plantvillage-leaf-map.json
│       └── plantdoc/{train,test}/
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    └── data/
        ├── download.py             # Kaggle, PlantDoc clone, leaf map
        ├── splits.py               # leaf-grouped 70/15/15, PlantDoc CSV
        ├── dataset.py              # Dataset, transforms, DataLoaders
        └── labels.py               # 38 class names and the crosswalk
```
