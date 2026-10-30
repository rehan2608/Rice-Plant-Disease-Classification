"""backend/service.py — Inference & Knowledge Service for Rice Disease Classification."""

import io
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

from src.features import extract_fused_features_for_image
from src.infer import get_model


# ─── Agronomic Disease Knowledge Base ────────────────────────────────
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "BL": {
        "code": "BL",
        "full_name": "Blast",
        "scientific_name": "Magnaporthe oryzae (syn. Pyricularia oryzae)",
        "pathogen_type": "Fungus",
        "severity_level": "High",
        "primary_symptoms": [
            "Diamond or spindle-shaped lesions with grayish-white centers and reddish-brown margins",
            "Rapid lesion coalescence leading to complete leaf blighting and canopy scorching",
            "Collar rot at leaf junction and neck blast causing completely sterile, bleached panicles",
        ],
        "favorable_conditions": (
            "Extended leaf wetness (>8 hours of dew), relative humidity >90%, moderate temperatures "
            "(20–28°C), and excessive nitrogen fertilization promoting succulent growth."
        ),
        "management": {
            "cultural": "Split nitrogen applications into 3–4 balanced doses; maintain continuous shallow flooding (3–5 cm) to suppress airborne conidia.",
            "chemical": "Foliar spray with Tricyclazole 75 WP (0.6 g/L) or Isoprothiolane (1.5 ml/L) at early tiller emergence.",
            "biological": "Seed treatment with Pseudomonas fluorescens or Bacillus subtilis antagonists.",
            "resistant_cultivars": "Plant blast-resistant cultivars (e.g., IR64, Tetep, or R-gene pyramided lines).",
        },
    },
    "BLB": {
        "code": "BLB",
        "full_name": "Bacterial Leaf Blight",
        "scientific_name": "Xanthomonas oryzae pv. oryzae",
        "pathogen_type": "Bacterium",
        "severity_level": "Critical",
        "primary_symptoms": [
            "Water-soaked to yellowish-green translucent lesions starting at leaf margins near blade tips",
            "Lesions expand downwards with distinctive wavy, undulated borders, turning bleached straw-colored",
            "Milky bacterial exudate beads ('bacterial ooze') drying into golden crusted beads along lesions",
            "Severe systemic seedling wilt ('Kresek') causing rolled grayish-green desiccated leaves",
        ],
        "favorable_conditions": (
            "Warm temperatures (25–34°C) with persistent humidity >70%, driving monsoonal rains, "
            "and deep standing floodwater facilitating bacterial dispersion."
        ),
        "management": {
            "cultural": "Avoid clipping seedling leaf tips during manual transplanting; practice intermittent field drainage and avoid flood overflow.",
            "chemical": "Foliar application of Copper Hydroxide (2 g/L) or Streptomycin sulphate + Tetracycline combinations.",
            "biological": "Apply bio-control agents such as Bacillus amyloliquefaciens or bacteriophage preparations.",
            "resistant_cultivars": "Deploy near-isogenic lines with multiple resistance genes (Xa4, xa5, Xa21).",
        },
    },
    "BS": {
        "code": "BS",
        "full_name": "Brown Spot",
        "scientific_name": "Bipolaris oryzae (syn. Helminthosporium oryzae)",
        "pathogen_type": "Fungus",
        "severity_level": "Moderate",
        "primary_symptoms": [
            "Numerous small, oval or circular brown spots evenly distributed across leaf blades",
            "Mature lesions develop lighter grayish-tan centers with pronounced dark brown rings",
            "Glume infection resulting in dark brown spotty discolored grains (pecky rice)",
        ],
        "favorable_conditions": (
            "High relative humidity (86–100%), ambient temperatures 16–36°C (optimum 25–30°C), "
            "nutrient-deficient or potassium/silicon-depleted soils, and drought spells."
        ),
        "management": {
            "cultural": "Apply balanced NPK fertilization with supplementary muriate of potash and silicon; avoid prolonged water stress.",
            "chemical": "Foliar fungicide application of Mancozeb (2.5 g/L) or Propiconazole (1 ml/L) at booting stage.",
            "biological": "Seed bio-priming with Trichoderma harzianum or Chaetomium globosum.",
            "resistant_cultivars": "Sow certified brown-spot tolerant cultivars suited for regional rainfed environments.",
        },
    },
    "SB": {
        "code": "SB",
        "full_name": "Sheath Blight",
        "scientific_name": "Rhizoctonia solani (teleomorph: Thanatephorus cucumeris)",
        "pathogen_type": "Fungus",
        "severity_level": "High",
        "primary_symptoms": [
            "Oval or oblong water-soaked, greenish-gray lesions initiated on lower leaf sheaths near waterline",
            "Lesions enlarge and merge with bleached grayish-white centers and dark brown margins ('snake skin' pattern)",
            "Ascending progression up sheaths to upper flag leaves causing premature lodging and empty panicles",
            "Globular dark brown sclerotia (1–3 mm) loosely adhering to lesions and floating in paddy water",
        ],
        "favorable_conditions": (
            "High canopy temperature (28–32°C), relative humidity >95%, dense crop canopy from "
            "narrow planting density, and excessive nitrogen top-dressing.",
        ),
        "management": {
            "cultural": "Adopt wider hill spacing (20 × 15 cm) for canopy aeration; skim floating sclerotia off water during puddling.",
            "chemical": "Foliar application of Hexaconazole 5% SC (2 ml/L) or Validamycin 3% L (2.5 ml/L) at maximum tillering.",
            "biological": "Incorporate Trichoderma viride or Pseudomonas fluorescens into soil and nursery.",
            "resistant_cultivars": "Utilize cultivars with open, erect plant architecture and moderate quantitative sheath blight tolerance.",
        },
    },
    "TU": {
        "code": "TU",
        "full_name": "Tungro",
        "scientific_name": "Rice tungro bacilliform virus (RTBV) & Rice tungro spherical virus (RTSV)",
        "pathogen_type": "Virus Complex",
        "severity_level": "Critical",
        "primary_symptoms": [
            "Distinct yellow to orange-yellow leaf discoloration originating at tips and advancing downward",
            "Marked plant stunting with shortened internodes, reduced tillering, and compact bushy stature",
            "Delayed heading with partially exerted, poorly filled panicles and blotched dark grains",
            "Younger emergent leaves displaying interveinal chlorotic mottling and slight leaf twisting",
        ],
        "favorable_conditions": (
            "High populations of Green Leafhopper vectors (Nephotettix virescens), asynchronous / "
            "staggered planting seasons providing continuous green bridges, and warm temperatures (25–30°C)."
        ),
        "management": {
            "cultural": "Enforce community-synchronized planting dates to disrupt leafhopper cycles; rogue and bury infected hills.",
            "chemical": "Target vector control with systemic insecticides like Thiamethoxam 25 WG (0.2 g/L) or Imidacloprid.",
            "biological": "Conserve predatory spider and mirid bug populations that naturally prey on leafhoppers.",
            "resistant_cultivars": "Cultivate green leafhopper-resistant or tungro-tolerant varieties (e.g., IR36, IR64).",
        },
    },
}


class RiceDiseaseService:
    """Service handling model inference, sample discovery, and agronomic knowledge retrieval."""

    def __init__(
        self,
        model_path: str = "outputs/final_model/rice_svm_model.pkl",
        dataset_base: str = "outputs/datasets/test",
    ):
        self.model_path = model_path
        self.dataset_base = dataset_base
        self._samples_cache: Optional[Dict[str, List[Dict[str, str]]]] = None

    def list_available_samples(self) -> Dict[str, List[Dict[str, str]]]:
        """Scan dataset directory for available sample images grouped by disease class."""
        if self._samples_cache is not None:
            return self._samples_cache

        base_path = Path(self.dataset_base)
        classes = ["BL", "BLB", "BS", "SB", "TU"]
        samples: Dict[str, List[Dict[str, str]]] = {c: [] for c in classes}

        if base_path.exists():
            for c in classes:
                cls_dir = base_path / c
                if cls_dir.is_dir():
                    files = sorted(
                        [
                            f.name
                            for f in cls_dir.iterdir()
                            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        ]
                    )
                    samples[c] = [{"filename": fname} for fname in files[:50]]

        self._samples_cache = samples
        return samples

    def get_sample_image_path(self, cls_name: str, filename: str) -> Optional[str]:
        """Resolve relative file path for a selected sample image."""
        p = Path(self.dataset_base) / cls_name / filename
        return str(p) if p.is_file() else None

    def _decode_image_input(self, image_input: Union[bytes, bytearray, io.BytesIO, str, Path, np.ndarray]) -> np.ndarray:
        """Decode diverse image input types into an RGB numpy uint8 array."""
        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            if image_input.ndim == 3:
                if image_input.shape[2] == 4:
                    return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                return image_input
            raise ValueError(f"Invalid image array shape: {image_input.shape}")

        if isinstance(image_input, io.BytesIO):
            image_input = image_input.getvalue()

        if isinstance(image_input, (bytes, bytearray)):
            buf = np.frombuffer(image_input, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Failed to decode image from raw bytes.")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {image_input}")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to read image from path: {image_input}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def classify_image(
        self,
        image_input: Union[bytes, bytearray, io.BytesIO, str, Path, np.ndarray],
        source_name: Optional[str] = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Perform full multi-level feature extraction and SVM inference."""
        t_total_start = time.perf_counter()

        try:
            # Step 1: Decode & Preprocessing
            t_pre_start = time.perf_counter()
            rgb_raw = self._decode_image_input(image_input)
            resized = cv2.resize(rgb_raw, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb_float = resized.astype(np.float32)
            enhanced = 2.0 * np.log1p(rgb_float)
            t_pre_end = time.perf_counter()

            # Step 2: 7,168-D Handcrafted Feature Extraction
            t_feat_start = time.perf_counter()
            fused_features = extract_fused_features_for_image(
                image_rgb=enhanced,
                level=3,
                cc_distance=1,
                l1_normalize_blocks=True,
            )
            t_feat_end = time.perf_counter()

            # Step 3: Model Inference & Softmax Scaling
            t_inf_start = time.perf_counter()
            clf = get_model(self.model_path)
            X = np.expand_dims(fused_features, axis=0)  # Shape (1, 7168)
            raw_scores = clf.decision_function(X)[0]
            classes = [str(c) for c in clf.classes_]

            # Softmax temperature scaling over decision margins
            scaled_scores = raw_scores / float(temperature)
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            probs = exp_scores / np.sum(exp_scores)
            prob_dict = {cls: float(round(p * 100.0, 1)) for cls, p in zip(classes, probs)}
            t_inf_end = time.perf_counter()

            # Top prediction & ranking
            pred_cls = max(prob_dict, key=prob_dict.get)
            meta = DISEASE_KNOWLEDGE_BASE.get(pred_cls, DISEASE_KNOWLEDGE_BASE["BL"])

            ranked_classes = []
            for cls in classes:
                cls_meta = DISEASE_KNOWLEDGE_BASE.get(cls, {})
                ranked_classes.append({
                    "class": cls,
                    "full_name": cls_meta.get("full_name", cls),
                    "confidence_percent": prob_dict[cls],
                    "decision_margin": float(raw_scores[classes.index(cls)]),
                })
            ranked_classes.sort(key=lambda item: item["confidence_percent"], reverse=True)

            t_total_end = time.perf_counter()

            return {
                "success": True,
                "source_name": source_name or "uploaded_image",
                "prediction": {
                    "class": pred_cls,
                    "full_name": meta["full_name"],
                    "scientific_name": meta["scientific_name"],
                    "pathogen_type": meta["pathogen_type"],
                    "severity_level": meta["severity_level"],
                    "confidence_percent": prob_dict[pred_cls],
                },
                "ranked_classes": ranked_classes,
                "disease_details": meta,
                "timings": {
                    "total_ms": round((t_total_end - t_total_start) * 1000, 1),
                    "preprocessing_ms": round((t_pre_end - t_pre_start) * 1000, 1),
                    "feature_extraction_ms": round((t_feat_end - t_feat_start) * 1000, 1),
                    "inference_ms": round((t_inf_end - t_inf_start) * 1000, 1),
                },
                "feature_specs": {
                    "total_dimensions": int(fused_features.shape[0]),
                    "correlogram_dim": 256,
                    "mclbp_dim": 6912,
                },
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "message": f"Classification error: {exc}",
            }


_GLOBAL_SERVICE: Optional[RiceDiseaseService] = None


def get_service() -> RiceDiseaseService:
    """Return singleton instance of RiceDiseaseService."""
    global _GLOBAL_SERVICE
    if _GLOBAL_SERVICE is None:
        _GLOBAL_SERVICE = RiceDiseaseService()
    return _GLOBAL_SERVICE
