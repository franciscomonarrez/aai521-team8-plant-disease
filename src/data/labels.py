"""PlantVillage class names and the PlantDoc crosswalk.

Folder names are kept exactly as they appear on disk, including the dataset's
own spellings (``Haunglongbing``, ``Common_rust_``) and the PlantDoc folder
``Soyabean leaf``.
"""

from __future__ import annotations

# Alphabetical on purpose: index 0..37 is this order, shared by every split.
PLANTVILLAGE_CLASSES: tuple[str, ...] = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)

CLASS_TO_IDX: dict[str, int] = {name: index for index, name in enumerate(PLANTVILLAGE_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {index: name for name, index in CLASS_TO_IDX.items()}
NUM_CLASSES = len(PLANTVILLAGE_CLASSES)

# (plantdoc folder, plantvillage folder, why this pair is the right match)
PLANTDOC_TO_PLANTVILLAGE: tuple[tuple[str, str, str], ...] = (
    (
        "Apple Scab Leaf",
        "Apple___Apple_scab",
        "Apple scab.",
    ),
    (
        "Apple leaf",
        "Apple___healthy",
        "Healthy apple leaf.",
    ),
    (
        "Apple rust leaf",
        "Apple___Cedar_apple_rust",
        "PlantDoc apple rust matches cedar-apple rust, the only apple rust class in PlantVillage.",
    ),
    (
        "Bell_pepper leaf",
        "Pepper,_bell___healthy",
        "Healthy bell pepper. PlantDoc uses an underscore in the folder name.",
    ),
    (
        "Bell_pepper leaf spot",
        "Pepper,_bell___Bacterial_spot",
        "Bell-pepper leaf spot is bacterial spot, PlantVillage's only pepper disease class.",
    ),
    (
        "Blueberry leaf",
        "Blueberry___healthy",
        "Healthy blueberry. PlantVillage has no diseased blueberry class.",
    ),
    (
        "Cherry leaf",
        "Cherry_(including_sour)___healthy",
        "Healthy cherry. PlantDoc has no cherry powdery-mildew class.",
    ),
    (
        "Corn Gray leaf spot",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Gray leaf spot (Cercospora).",
    ),
    (
        "Corn leaf blight",
        "Corn_(maize)___Northern_Leaf_Blight",
        "PlantDoc corn blight maps to northern leaf blight, not to gray leaf spot (that class is separate).",
    ),
    (
        "Corn rust leaf",
        "Corn_(maize)___Common_rust_",
        "Common rust. The PlantVillage folder name keeps a trailing underscore.",
    ),
    (
        "Peach leaf",
        "Peach___healthy",
        "Healthy peach. PlantDoc has no peach bacterial-spot class.",
    ),
    (
        "Potato leaf early blight",
        "Potato___Early_blight",
        "Early blight.",
    ),
    (
        "Potato leaf late blight",
        "Potato___Late_blight",
        "Late blight.",
    ),
    (
        "Raspberry leaf",
        "Raspberry___healthy",
        "Healthy raspberry.",
    ),
    (
        "Soyabean leaf",
        "Soybean___healthy",
        "Healthy soybean. PlantDoc spells the folder Soyabean.",
    ),
    (
        "Squash Powdery mildew leaf",
        "Squash___Powdery_mildew",
        "Powdery mildew.",
    ),
    (
        "Strawberry leaf",
        "Strawberry___healthy",
        "Healthy strawberry. This is not leaf scorch.",
    ),
    (
        "Tomato Early blight leaf",
        "Tomato___Early_blight",
        "Early blight.",
    ),
    (
        "Tomato Septoria leaf spot",
        "Tomato___Septoria_leaf_spot",
        "Septoria leaf spot.",
    ),
    (
        "Tomato leaf",
        "Tomato___healthy",
        "Healthy tomato.",
    ),
    (
        "Tomato leaf bacterial spot",
        "Tomato___Bacterial_spot",
        "Bacterial spot.",
    ),
    (
        "Tomato leaf late blight",
        "Tomato___Late_blight",
        "Late blight.",
    ),
    (
        "Tomato leaf mosaic virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato mosaic virus.",
    ),
    (
        "Tomato leaf yellow virus",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Yellow-virus symptoms map to tomato yellow leaf curl virus.",
    ),
    (
        "Tomato mold leaf",
        "Tomato___Leaf_Mold",
        "Leaf mold.",
    ),
    (
        "Tomato two spotted spider mites leaf",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Two-spotted spider mite. Upstream PlantDoc keeps this class in train only.",
    ),
    (
        "grape leaf",
        "Grape___healthy",
        "Healthy grape. The PlantDoc folder name is lowercase.",
    ),
    (
        "grape leaf black rot",
        "Grape___Black_rot",
        "Black rot. The PlantDoc folder name is lowercase.",
    ),
)

# PlantVillage classes with no honest PlantDoc counterpart. Left out of the
# field-photo test CSV rather than forced onto a nearby disease.
UNMAPPED_PLANTVILLAGE: tuple[tuple[str, str], ...] = (
    (
        "Apple___Black_rot",
        "No apple black-rot class in PlantDoc.",
    ),
    (
        "Cherry_(including_sour)___Powdery_mildew",
        "PlantDoc cherry images are healthy leaves only.",
    ),
    (
        "Corn_(maize)___healthy",
        "PlantDoc has diseased corn leaves but no healthy corn class.",
    ),
    (
        "Grape___Esca_(Black_Measles)",
        "No esca / black-measles class in PlantDoc.",
    ),
    (
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "No grape leaf-blight class in PlantDoc.",
    ),
    (
        "Orange___Haunglongbing_(Citrus_greening)",
        "PlantDoc does not include citrus. Spelling Haunglongbing is the dataset's.",
    ),
    (
        "Peach___Bacterial_spot",
        "PlantDoc peach images are healthy leaves only.",
    ),
    (
        "Potato___healthy",
        "PlantDoc has early and late blight but no healthy potato class.",
    ),
    (
        "Strawberry___Leaf_scorch",
        "PlantDoc strawberry images are healthy leaves, not leaf scorch.",
    ),
    (
        "Tomato___Target_Spot",
        "No tomato target-spot class in PlantDoc.",
    ),
)

PLANTDOC_TO_PLANTVILLAGE_LABEL: dict[str, str] = {
    plantdoc_class: plantvillage_label
    for plantdoc_class, plantvillage_label, _notes in PLANTDOC_TO_PLANTVILLAGE
}

CROP_DISPLAY_NAMES: dict[str, str] = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper,_bell": "Bell pepper",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}


def crop_and_condition(label: str) -> tuple[str, str]:
    """Split a PlantVillage folder name into ``(crop, condition)``."""
    crop, condition = label.split("___", 1)
    return crop, condition


def display_crop(label: str) -> str:
    """Short crop name for charts (``Pepper,_bell`` becomes ``Bell pepper``)."""
    crop, _condition = crop_and_condition(label)
    return CROP_DISPLAY_NAMES.get(crop, crop)


def is_healthy(label: str) -> bool:
    """True when the PlantVillage condition is the healthy class."""
    _crop, condition = crop_and_condition(label)
    return condition.lower() == "healthy"


def _check_label_tables() -> None:
    if tuple(sorted(PLANTVILLAGE_CLASSES)) != PLANTVILLAGE_CLASSES:
        raise RuntimeError("PLANTVILLAGE_CLASSES must stay in alphabetical order.")
    if len(PLANTVILLAGE_CLASSES) != len(set(PLANTVILLAGE_CLASSES)):
        raise RuntimeError("Duplicate PlantVillage class name.")
    mapped = [plantvillage for _plantdoc, plantvillage, _notes in PLANTDOC_TO_PLANTVILLAGE]
    plantdoc_names = [plantdoc for plantdoc, _plantvillage, _notes in PLANTDOC_TO_PLANTVILLAGE]
    unmapped = [plantvillage for plantvillage, _notes in UNMAPPED_PLANTVILLAGE]
    if len(mapped) != len(set(mapped)):
        raise RuntimeError("Duplicate PlantVillage label in the PlantDoc map.")
    if len(plantdoc_names) != len(set(plantdoc_names)):
        raise RuntimeError("Duplicate PlantDoc class in the map.")
    if len(unmapped) != len(set(unmapped)):
        raise RuntimeError("Duplicate unmapped PlantVillage class.")
    if set(mapped) & set(unmapped):
        raise RuntimeError("A PlantVillage class is both mapped and unmapped.")
    covered = set(mapped) | set(unmapped)
    if covered != set(PLANTVILLAGE_CLASSES):
        missing = set(PLANTVILLAGE_CLASSES) - covered
        extra = covered - set(PLANTVILLAGE_CLASSES)
        raise RuntimeError(f"PlantDoc map is not a partition of PlantVillage. missing={missing} extra={extra}")


_check_label_tables()
