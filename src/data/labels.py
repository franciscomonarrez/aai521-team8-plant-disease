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

# (plantdoc folder, plantvillage folder, confidence, notes). "approximate"
# means the PlantDoc folder is broader than the PlantVillage class.
PLANTDOC_TO_PLANTVILLAGE: tuple[tuple[str, str, str, str], ...] = (
    (
        "Apple Scab Leaf",
        "Apple___Apple_scab",
        "exact",
        "Apple scab.",
    ),
    (
        "Apple leaf",
        "Apple___healthy",
        "exact",
        "Healthy apple leaf. Names that say yellowing or aphid leaf curl "
        "are excluded.",
    ),
    (
        "Apple rust leaf",
        "Apple___Cedar_apple_rust",
        "approximate",
        "PlantDoc says only 'apple rust'; cedar-apple rust is PlantVillage's "
        "one apple rust class. Names that say scab, black rot or rust mite "
        "are excluded.",
    ),
    (
        "Bell_pepper leaf",
        "Pepper,_bell___healthy",
        "exact",
        "Healthy bell pepper (the folder name has an underscore). Names that "
        "say yellow or curling leaves, a virus or pests are excluded.",
    ),
    (
        "Bell_pepper leaf spot",
        "Pepper,_bell___Bacterial_spot",
        "approximate",
        "PlantDoc 'leaf spot' is broader than bacterial spot, PlantVillage's "
        "only pepper disease. Names that say canker, gray leaf spot, "
        "Septoria or dogwood are excluded.",
    ),
    (
        "Blueberry leaf",
        "Blueberry___healthy",
        "exact",
        "Healthy blueberry; PlantVillage has no diseased blueberry class. "
        "Names that say disease, damage, chlorosis, iron deficiency or a "
        "pest are excluded.",
    ),
    (
        "Cherry leaf",
        "Cherry_(including_sour)___healthy",
        "exact",
        "Healthy cherry; PlantDoc has no cherry powdery-mildew class. Names "
        "that say nitrogen deficiency or cherry laurel are excluded.",
    ),
    (
        "Corn Gray leaf spot",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "exact",
        "Gray leaf spot (Cercospora). Two comparison figures (with northern "
        "leaf blight, with bacterial leaf streak) are excluded.",
    ),
    (
        "Corn leaf blight",
        "Corn_(maize)___Northern_Leaf_Blight",
        "approximate",
        "PlantDoc's leaf-blight folder mixes corn diseases. Names that say "
        "gray leaf spot, bacterial leaf streak, southern corn leaf blight, "
        "Goss's wilt, Stewart's wilt, anthracnose or leaf scald are "
        "excluded; unnamed photos of those diseases may remain.",
    ),
    (
        "Corn rust leaf",
        "Corn_(maize)___Common_rust_",
        "approximate",
        "PlantDoc says only 'corn rust'. Names that say southern rust "
        "(Puccinia polysora, a different pathogen), stripe rust or tar spot "
        "are excluded; unnamed southern-rust photos may remain. The "
        "PlantVillage folder name keeps a trailing underscore.",
    ),
    (
        "Peach leaf",
        "Peach___healthy",
        "exact",
        "Healthy peach; PlantDoc has no peach bacterial-spot class. A photo "
        "named as a rose tree is excluded.",
    ),
    (
        "Potato leaf early blight",
        "Potato___Early_blight",
        "exact",
        "Early blight. Many names in this folder say late blight "
        "(Phytophthora infestans) or early blight on tomato; those are "
        "excluded.",
    ),
    (
        "Potato leaf late blight",
        "Potato___Late_blight",
        "exact",
        "Late blight. Names that also say early blight are excluded.",
    ),
    (
        "Raspberry leaf",
        "Raspberry___healthy",
        "exact",
        "Healthy raspberry. Names that say rust, spots, beetles, iron "
        "deficiency or poison ivy are excluded.",
    ),
    (
        "Soyabean leaf",
        "Soybean___healthy",
        "exact",
        "Healthy soybean (PlantDoc spells it Soyabean). Names that say "
        "potassium deficiency, brown spot, downy mildew or raspberry are "
        "excluded.",
    ),
    (
        "Squash Powdery mildew leaf",
        "Squash___Powdery_mildew",
        "exact",
        "Powdery mildew. Names that say cucumber or downy mildew are "
        "excluded.",
    ),
    (
        "Strawberry leaf",
        "Strawberry___healthy",
        "exact",
        "Healthy strawberry, not leaf scorch. Names that say red spots, "
        "yellow leaves or cold damage are excluded.",
    ),
    (
        "Tomato Early blight leaf",
        "Tomato___Early_blight",
        "exact",
        "Early blight. Names that say potato, canker or late blight are "
        "excluded.",
    ),
    (
        "Tomato Septoria leaf spot",
        "Tomato___Septoria_leaf_spot",
        "exact",
        "Septoria leaf spot. Names that say early blight, gray leaf spot, "
        "bacterial spot or bean rust are excluded.",
    ),
    (
        "Tomato leaf",
        "Tomato___healthy",
        "exact",
        "Healthy tomato. Names that say late blight, bacterial leaf spot, "
        "curling, yellow leaves or basil are excluded.",
    ),
    (
        "Tomato leaf bacterial spot",
        "Tomato___Bacterial_spot",
        "exact",
        "Bacterial spot (Xanthomonas). Names that say bacterial speck or "
        "canker, which are different bacterial diseases, are excluded.",
    ),
    (
        "Tomato leaf late blight",
        "Tomato___Late_blight",
        "exact",
        "Late blight. A name that says early blight is excluded.",
    ),
    (
        "Tomato leaf mosaic virus",
        "Tomato___Tomato_mosaic_virus",
        "approximate",
        "PlantDoc says only 'mosaic virus'; PlantVillage's class is tomato "
        "mosaic virus (ToMV). Names that say another virus (ToBRFV; TMV, "
        "CMV or PepMV; leaf curl on luffa) are excluded.",
    ),
    (
        "Tomato leaf yellow virus",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "approximate",
        "Yellow-virus symptoms map to tomato yellow leaf curl virus, the "
        "nearest class. A name that also says mosaic virus is excluded.",
    ),
    (
        "Tomato mold leaf",
        "Tomato___Leaf_Mold",
        "exact",
        "Leaf mold. Names that say gray leaf spot or TYLC are excluded.",
    ),
    (
        "Tomato two spotted spider mites leaf",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "exact",
        "Both PlantDoc images are labeled comparison panels (spot, speck, "
        "early blight and mite damage; canker), not spider-mite photos. "
        "Both are excluded, so this class has no field-test images.",
    ),
    (
        "grape leaf",
        "Grape___healthy",
        "exact",
        "Healthy grape. The PlantDoc folder name is lowercase.",
    ),
    (
        "grape leaf black rot",
        "Grape___Black_rot",
        "exact",
        "Black rot. The PlantDoc folder name is lowercase. Names that say "
        "downy mildew, anthracnose or rust are excluded.",
    ),
)

# PlantDoc folders whose images were all excluded.
PLANTDOC_WITHOUT_USABLE_IMAGES: tuple[str, ...] = (
    "Tomato two spotted spider mites leaf",
)

# PlantVillage classes with no PlantDoc counterpart.
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
    for plantdoc_class, plantvillage_label, _confidence, _notes in (
        PLANTDOC_TO_PLANTVILLAGE
    )
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
    mapped = [row[1] for row in PLANTDOC_TO_PLANTVILLAGE]
    plantdoc_names = [row[0] for row in PLANTDOC_TO_PLANTVILLAGE]
    if {row[2] for row in PLANTDOC_TO_PLANTVILLAGE} - {"exact", "approximate"}:
        raise RuntimeError("PlantDoc map confidence must be exact or approximate.")
    if set(PLANTDOC_WITHOUT_USABLE_IMAGES) - set(plantdoc_names):
        raise RuntimeError("PLANTDOC_WITHOUT_USABLE_IMAGES names an unknown folder.")
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
