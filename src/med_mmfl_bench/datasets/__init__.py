"""Dataset implementations for multimodal medical data.

Provides PyTorch Dataset classes for:
    - **SYMILE-MIMIC**: 3-modality contrastive learning (CXR + ECG + Labs)
    - **MIMIC-CXR**: Image-text multi-label classification
    - **BraTS**: 3D brain tumor segmentation
    - **EHRXQA**: EHR-based visual question answering
    - **PathVQA**: Pathology visual question answering (Yes/No)
"""

from med_mmfl_bench.datasets.symile_mimic import (
    SymileMIMICDataset,
    SymileMIMICRetrievalDataset,
)

__all__ = [
    "SymileMIMICDataset",
    "SymileMIMICRetrievalDataset",
]

# Conditional imports — these datasets have heavier dependencies
# (transformers, nibabel, etc.) that may not always be installed.

try:
    from med_mmfl_bench.datasets.mimic_cxr import MimicMultiModal, MimicPublic

    __all__.extend(["MimicMultiModal", "MimicPublic"])
except ImportError:
    pass

try:
    from med_mmfl_bench.datasets.brats import BraTS24GLIPostDataset

    __all__.append("BraTS24GLIPostDataset")
except ImportError:
    pass

try:
    from med_mmfl_bench.datasets.ehrxqa import EHRXQA

    __all__.append("EHRXQA")
except ImportError:
    pass

try:
    from med_mmfl_bench.datasets.pathvqa import YesNoVQADataset

    __all__.append("YesNoVQADataset")
except ImportError:
    pass
