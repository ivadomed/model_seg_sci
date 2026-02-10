"""
Extract scanner information and sagittal T2w voxel dimensions from MRI metadata.

This script:
- reads CSV file with participant_id and session_id columns
- recursively finds sagittal T2w images for each participant/session across sci-zurich and nisci-trial datasets
- extracts scanner metadata from sagittal images (Manufacturer, ManufacturersModelName, MagneticFieldStrength)
- extracts voxel dimensions (pixdim1, pixdim2, pixdim3) from sagittal images using fslinfo
- aggregates the information into a dataframe and saves to CSV
- prints basic summary statistics

Example usage:
    python 06_get_scanner_info.py \
        -i <PATH_TO_CSV_FILE> \
        -d <PATH_TO_DATA_ROOT_CONTAINING_DATASETS> \
        -o <OUTPUT_CSV>
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)


# Default location of nisci-trial participants.tsv; can be overridden via CLI
PARTICIPANTS_TSV_NISCI_DEFAULT = Path(
    os.path.expanduser("~/data/data.neuro.polymtl.ca/nisci-trial/participants.tsv")
)


def get_parser() -> argparse.ArgumentParser:
    """Parser function for command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract scanner information from sagittal T2w MRI metadata.",
        prog="get_scanner_info_sagittal",
    )
    parser.add_argument(
        "-i",
        "--input-csv",
        required=True,
        type=str,
        help="Absolute path to a CSV file containing participant_id and session_id columns.",
    )
    parser.add_argument(
        "-d",
        "--data-root",
        required=True,
        type=str,
        help="Path to the directory containing the sci-zurich and nisci-trial datasets.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        required=True,
        type=str,
        help="Path to the output CSV file for saving results.",
    )
    parser.add_argument(
        "--nisci-participants-tsv",
        type=str,
        default=str(PARTICIPANTS_TSV_NISCI_DEFAULT),
        help=(
            "Path to nisci-trial participants.tsv used to obtain MagneticFieldStrength "
            "for nisci-trial subjects when JSON sidecars are empty. "
            f"Default: {PARTICIPANTS_TSV_NISCI_DEFAULT}"
        ),
    )
    return parser


def get_nisci_field_strengths(participants_tsv: Path) -> Dict[str, float]:
    """Load MagneticFieldStrength per participant from nisci-trial participants.tsv.

    Returns a mapping from participant_id (e.g. 'sub-001') to MagneticFieldStrength.
    If the file is missing or unreadable, returns an empty dict.
    """
    field_map: Dict[str, float] = {}
    if not participants_tsv.exists():
        logging.warning("nisci-trial participants.tsv not found at %s", participants_tsv)
        return field_map

    try:
        df_part = pd.read_csv(participants_tsv, sep="\t")
    except Exception as e:  # noqa: BLE001
        logging.warning("Could not read nisci-trial participants.tsv: %s", e)
        return field_map

    if "participant_id" not in df_part.columns or "MagneticFieldStrength" not in df_part.columns:
        logging.warning("participants.tsv missing required columns 'participant_id' and/or 'MagneticFieldStrength'")
        return field_map

    for _, row in df_part.iterrows():
        pid = str(row["participant_id"]).strip()
        try:
            fs = float(row["MagneticFieldStrength"])
        except Exception:  # noqa: BLE001
            continue
        if pid:
            field_map[pid] = fs

    return field_map


def load_input_table(csv_path: Path) -> pd.DataFrame:
    """Load input CSV and validate required columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"participant_id", "session_id"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV file must contain columns: {', '.join(sorted(required_cols))}. Missing: {', '.join(sorted(missing))}")

    # Basic cleanup
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["session_id"] = df["session_id"].astype(str).str.strip()

    # For sci-nisci, if session_id is empty, default to 'ses-01' to ease file lookup
    df.loc[df["session_id"].isin(["", "nan", "None"]), "session_id"] = "ses-01"

    return df


def build_dataset_roots(data_root: Path) -> Dict[str, Path]:
    """Return existing dataset roots under the provided data_root."""
    datasets = {}
    for name in ("sci-zurich", "nisci-trial"):
        ds_path = data_root / name
        if ds_path.is_dir():
            datasets[name] = ds_path
        else:
            logging.warning("Dataset directory not found under data_root: %s", ds_path)
    if not datasets:
        logging.error("No dataset directories (sci-zurich or nisci-trial) found under %s", data_root)
    return datasets


def find_sagittal_t2w_image(
    participant_id: str,
    session_id: str,
    dataset_roots: Dict[str, Path],
) -> Optional[Dict[str, object]]:
    """Find sagittal T2w image for a given participant/session across datasets.

    Strategy (simplified per user request):
    - Build BIDS-like path <dataset_root>/sub-XXX/ses-YYY/anat
    - Look for a single '*ses-*_acq-sag_T2w.nii.gz' file under that anat folder
      (non-recursive glob).
    - If found, return it; if multiple, pick the first sorted path and log a warning.
    - If none found in all datasets, return None.
    """
    # Normalize to possible BIDS-like IDs if needed
    sub_id = participant_id if participant_id.startswith("sub-") else f"sub-{participant_id}"

    # Normalize session: allow bare '01' or empty to map to 'ses-01'
    if session_id.startswith("ses-"):
        ses_id = session_id
    elif session_id in ("", "nan", "None"):
        ses_id = "ses-01"
    else:
        ses_id = f"ses-{session_id}"

    for ds_name, ds_root in dataset_roots.items():
        anat = ds_root / sub_id / ses_id / "anat"
        if not anat.is_dir():
            continue

        # Strict pattern: *ses-*_acq-sag_T2w.nii.gz
        pattern = f"*{ses_id}_acq-sag_T2w.nii.gz"
        candidates = sorted(anat.glob(pattern))

        if not candidates:
            continue

        if len(candidates) > 1:
            logging.warning(
                "Multiple sagittal T2w files matched pattern %s for %s %s in dataset %s. Choosing %s.",
                pattern,
                participant_id,
                ses_id,
                ds_name,
                candidates[0],
            )

        nifti_path = candidates[0]
        json_path = nifti_path.with_suffix("")  # drop .gz if present
        if json_path.name.endswith(".nii"):
            json_path = json_path.with_suffix("")
        json_path = json_path.with_suffix(".json")

        return {
            "dataset": ds_name,
            "nifti_path": nifti_path,
            "json_path": json_path if json_path.exists() else None,
            "search_status": "found",
        }

    logging.warning(
        "Sagittal T2w image not found for participant_id=%s, session_id=%s",
        participant_id,
        session_id,
    )
    return None


def extract_metadata_from_json(json_path: Optional[Path]) -> Dict[str, Optional[object]]:
    """Extract scanner metadata from JSON sidecar file.

    Handles both simple BIDS-style JSONs and sci-zurich JSONs where
    scanner info is nested under the first element of the ``acqpar`` list.
    """
    metadata = {
        "Manufacturer": None,
        "ManufacturersModelName": None,
        "MagneticFieldStrength": None,
    }

    if json_path is None:
        logging.warning("JSON sidecar not found (json_path is None)")
        return metadata

    if not json_path.exists():
        logging.warning("JSON sidecar file does not exist: %s", json_path)
        return metadata

    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:  # noqa: BLE001
        logging.warning("Error reading JSON file %s: %s", json_path, e)
        return metadata

    def _clean_str(val: object) -> object:
        """Strip trailing/leading whitespace from strings, leave others unchanged."""
        if isinstance(val, str):
            return val.strip()
        return val

    # 1) Try top-level keys first (BIDS-like JSON)
    manufacturer = (
        json_data.get("Manufacturer")
        or json_data.get("Manufacturer ")
        or json_data.get("manufacturer")
    )
    model = (
        json_data.get("ManufacturersModelName")
        or json_data.get("ManufacturerModelName")
        or json_data.get("ManufacturersModelName ")
        or json_data.get("manufacturer_model_name")
    )
    field = json_data.get("MagneticFieldStrength")

    # 2) Fallback: sci-zurich JSON with data in acqpar[0]
    if (manufacturer is None or model is None or field is None) and "acqpar" in json_data:
        try:
            acq_list = json_data.get("acqpar")
            if isinstance(acq_list, list) and acq_list:
                acq0 = acq_list[0]
                if isinstance(acq0, dict):
                    manufacturer = manufacturer or acq0.get("Manufacturer")
                    model = (
                        model
                        or acq0.get("ManufacturersModelName")
                        or acq0.get("ManufacturerModelName")
                    )
                    field = field or acq0.get("MagneticFieldStrength")
        except Exception as e:  # noqa: BLE001
            logging.warning("Error accessing 'acqpar' in JSON file %s: %s", json_path, e)

    manufacturer = _clean_str(manufacturer)
    model = _clean_str(model)

    metadata["Manufacturer"] = manufacturer
    metadata["ManufacturersModelName"] = model
    metadata["MagneticFieldStrength"] = field

    return metadata


def extract_pixdim_from_nifti(
    nifti_path: Optional[Path],
    fslinfo_available: bool,
) -> Dict[str, Optional[float]]:
    """Extract voxel dimensions (pixdim1, pixdim2, pixdim3) using fslinfo."""
    pixdim = {"pixdim1": None, "pixdim2": None, "pixdim3": None}

    if nifti_path is None:
        return pixdim

    if not fslinfo_available:
        return pixdim

    try:
        result = subprocess.run(  # noqa: S603
            ["fslinfo", str(nifti_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        logging.warning("fslinfo command not found during execution.")
        return pixdim
    except subprocess.TimeoutExpired:
        logging.warning("fslinfo timed out for %s", nifti_path)
        return pixdim
    except Exception as e:  # noqa: BLE001
        logging.warning("Error running fslinfo for %s: %s", nifti_path, e)
        return pixdim

    if result.returncode != 0:
        logging.warning("fslinfo failed for %s: %s", nifti_path, result.stderr.strip())
        return pixdim

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("pixdim1"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pixdim["pixdim1"] = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith("pixdim2"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pixdim["pixdim2"] = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith("pixdim3"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pixdim["pixdim3"] = float(parts[1])
                except ValueError:
                    pass

    return pixdim


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print basic summary statistics of the scanner information."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal records: {len(df)}")

    n_found = df["sag_t2w_path"].notna().sum()
    print(f"Records with sagittal T2w image found: {n_found} ({n_found / len(df) * 100:.1f}%)")

    if "dataset" in df.columns:
        print("\n--- Dataset Distribution ---")
        print(df["dataset"].value_counts(dropna=False))

    if "Manufacturer" in df.columns:
        print("\n--- Manufacturer Distribution ---")
        print(df["Manufacturer"].value_counts(dropna=False))

    if "ManufacturersModelName" in df.columns:
        print("\n--- Scanner Model Distribution ---")
        print(df["ManufacturersModelName"].value_counts(dropna=False))

    if "MagneticFieldStrength" in df.columns:
        print("\n--- Magnetic Field Strength Distribution ---")
        print(df["MagneticFieldStrength"].value_counts(dropna=False).sort_index())

    print("\n--- Voxel Dimensions (pixdim) Statistics (sagittal T2w) ---")
    for dim in ["pixdim1", "pixdim2", "pixdim3"]:
        if dim in df.columns:
            valid = df[dim].dropna()
            if len(valid) > 0:
                print(f"{dim}: mean={valid.mean():.4f}, sd={valid.std():.4f}, min={valid.min():.4f}, max={valid.max():.4f}, N={len(valid)}")

    print("\n--- Missing Data ---")
    for col in [
        "Manufacturer",
        "ManufacturersModelName",
        "MagneticFieldStrength",
        "pixdim1",
        "pixdim2",
        "pixdim3",
    ]:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"{col}: {missing} missing ({missing / len(df) * 100:.1f}%)")

    print("\n" + "=" * 80)


def print_publication_ready_resolution(df: pd.DataFrame) -> None:
    """Print publication-ready resolution sentences per dataset.

    Format: min-max (mean ± std) for in-plane resolution and slice thickness
    separately for each dataset.
    """
    if "dataset" not in df.columns:
        return

    print("\n" + "=" * 80)
    print("PUBLICATION-READY RESOLUTION (SAGITTAL T2w)")
    print("=" * 80)

    # Only keep rows with voxel dimensions
    if not {"pixdim1", "pixdim2", "pixdim3"}.issubset(df.columns):
        print("No voxel dimension columns available to compute resolution summaries.")
        print("\n" + "=" * 80)
        return

    for dataset, df_ds in df.groupby("dataset", dropna=True):
        # Drop rows with missing any of the pixdims
        valid = df_ds[["pixdim1", "pixdim2", "pixdim3"]].dropna()
        if valid.empty:
            continue

        # In-plane resolution: mean of pixdim1 and pixdim2
        inplane = (valid["pixdim1"] + valid["pixdim2"]) / 2.0
        thickness = valid["pixdim3"]

        inplane_min, inplane_max = inplane.min(), inplane.max()
        inplane_mean, inplane_std = inplane.mean(), inplane.std()

        thick_min, thick_max = thickness.min(), thickness.max()
        thick_mean, thick_std = thickness.mean(), thickness.std()

        sentence = (
            f"For {dataset}, the sagittal T2w in-plane resolution ranged from "
            f"{inplane_min:.2f}-{inplane_max:.2f} mm ({inplane_mean:.2f} ± {inplane_std:.2f} mm), "
            f"and the slice thickness ranged from {thick_min:.2f}-{thick_max:.2f} mm "
            f"({thick_mean:.2f} ± {thick_std:.2f} mm)."
        )
        print("\n" + sentence)

    print("\n" + "=" * 80)


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    input_csv = Path(os.path.expanduser(args.input_csv))
    data_root = Path(os.path.expanduser(args.data_root))
    output_csv = Path(os.path.expanduser(args.output_csv))
    nisci_participants_tsv = Path(os.path.expanduser(args.nisci_participants_tsv))

    logging.info("Reading input table from: %s", input_csv)
    df_input = load_input_table(input_csv)

    logging.info("Using data root: %s", data_root)
    dataset_roots = build_dataset_roots(data_root)
    if not dataset_roots:
        logging.error("No valid datasets found under data_root. Exiting.")
        return

    # Pre-load MagneticFieldStrength from nisci-trial participants.tsv
    nisci_field_strengths = get_nisci_field_strengths(nisci_participants_tsv)

    fslinfo_available = shutil.which("fslinfo") is not None
    if not fslinfo_available:
        logging.warning("fslinfo not found in PATH. Voxel dimensions will be left as NaN.")

    results = []

    total = len(df_input)
    logging.info("Processing %d records...", total)

    for idx, row in df_input.iterrows():
        participant_id = str(row["participant_id"])
        session_id = str(row["session_id"])

        if (idx + 1) % 20 == 0 or idx == 0:
            logging.info("  Processed %d/%d records...", idx + 1, total)

        try:
            found = find_sagittal_t2w_image(participant_id, session_id, dataset_roots)
            if found is None:
                results.append(
                    {
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "dataset": None,
                        "sag_t2w_path": None,
                        "sag_t2w_json_path": None,
                        "search_status": "not_found",
                        "Manufacturer": None,
                        "ManufacturersModelName": None,
                        "MagneticFieldStrength": None,
                        "pixdim1": None,
                        "pixdim2": None,
                        "pixdim3": None,
                    }
                )
                continue

            nifti_path = found["nifti_path"]
            json_path = found["json_path"]

            metadata = extract_metadata_from_json(json_path)

            # For nisci-trial, JSON sidecars may be empty; override MagneticFieldStrength
            if not metadata["MagneticFieldStrength"] and found["dataset"] == "nisci-trial":
                # Ensure participant_id key has 'sub-' prefix as in participants.tsv
                pid_key = participant_id if participant_id.startswith("sub-") else f"sub-{participant_id}"
                if pid_key in nisci_field_strengths:
                    metadata["MagneticFieldStrength"] = nisci_field_strengths[pid_key]

            pixdim = extract_pixdim_from_nifti(nifti_path, fslinfo_available=fslinfo_available)

            results.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "dataset": found["dataset"],
                    "sag_t2w_path": str(nifti_path) if nifti_path is not None else None,
                    "sag_t2w_json_path": str(json_path) if json_path is not None else None,
                    "search_status": found.get("search_status", "found"),
                    "Manufacturer": metadata["Manufacturer"],
                    "ManufacturersModelName": metadata["ManufacturersModelName"],
                    "MagneticFieldStrength": metadata["MagneticFieldStrength"],
                    "pixdim1": pixdim["pixdim1"],
                    "pixdim2": pixdim["pixdim2"],
                    "pixdim3": pixdim["pixdim3"],
                }
            )
        except Exception as e:  # noqa: BLE001
            logging.error(
                "Unexpected error for participant_id=%s, session_id=%s: %s",
                participant_id,
                session_id,
                e,
            )
            results.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "dataset": None,
                    "sag_t2w_path": None,
                    "sag_t2w_json_path": None,
                    "search_status": "error",
                    "Manufacturer": None,
                    "ManufacturersModelName": None,
                    "MagneticFieldStrength": None,
                    "pixdim1": None,
                    "pixdim2": None,
                    "pixdim3": None,
                }
            )

    df_results = pd.DataFrame(results)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    logging.info("Results saved to: %s", output_csv)

    print_summary_statistics(df_results)
    print_publication_ready_resolution(df_results)


if __name__ == "__main__":  # pragma: no cover
    main()

