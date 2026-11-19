"""Copy the nine selected nonlinear runs and split their deformed-state maps."""

import csv
from pathlib import Path
import shutil

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
CASE_DIRS = ("outputs", "outputs_barrel", "outputs_dome")
DESTINATION = ROOT / "outputs_9_soluzioni_ottimali"


def split_deformed_map(source, plastic_path, instability_path):
    image = Image.open(source).convert("RGBA")
    pixels = np.asarray(image).copy()
    rgb = pixels[:, :, :3].astype(np.int16)
    red, green, blue = (rgb[:, :, channel] for channel in range(3))

    grayscale = (np.maximum.reduce((red, green, blue)) < 245) & (
        np.maximum.reduce((red, green, blue))
        - np.minimum.reduce((red, green, blue)) < 22
    )
    plastic_onset = (blue > red + 30) & (blue > green + 30) & (blue > 50)
    plastic_full = (red > green + 30) & (red > blue + 30) & (green < red * 0.45)
    plastic = plastic_onset | plastic_full
    instability = (red > green + 20) & (green > blue + 20) & (red > 90)

    for path, mask in (
        (plastic_path, grayscale | plastic),
        (instability_path, grayscale | instability),
    ):
        result = pixels.copy()
        result[:, :, 3] = np.where(mask, pixels[:, :, 3], 0).astype(np.uint8)
        Image.fromarray(result, mode="RGBA").save(path)


def selected_runs():
    selected = []
    for case_dir in CASE_DIRS:
        summary_path = ROOT / case_dir / "optimal_section_models" / "soluzioni_ottimali.csv"
        with summary_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream, delimiter=";")
            for row in rows:
                candidate_id = row["candidate_id"]
                matches = list(
                    (ROOT / case_dir / "pareto_nonlinear_runs").glob(
                        f"run_*_reused_*_candidate_{candidate_id}"
                    )
                )
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected one nonlinear run for {case_dir}/{candidate_id}, found {len(matches)}"
                    )
                if not (matches[0] / "deformed_shape.png").is_file():
                    raise FileNotFoundError(f"Missing deformed_shape.png in {matches[0]}")
                selected.append((case_dir, row["optimum"], candidate_id, matches[0]))
    if len(selected) != 9:
        raise ValueError(f"Expected nine optimal runs, found {len(selected)}")
    return selected


def main():
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {DESTINATION}")
    runs = selected_runs()
    DESTINATION.mkdir()
    manifest = []

    for case_dir, optimum, candidate_id, source in runs:
        case_name = case_dir.removeprefix("outputs_")
        target = DESTINATION / case_name / f"{optimum}_candidate_{candidate_id}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        split_deformed_map(
            target / "deformed_shape.png",
            target / "mappa_plasticizzazioni_deformata.png",
            target / "mappa_instabilita_locale_deformata.png",
        )
        manifest.append(
            {
                "case": case_name,
                "optimum": optimum,
                "candidate_id": candidate_id,
                "source": str(source.relative_to(ROOT)),
                "copy": str(target.relative_to(DESTINATION)),
            }
        )
        print(f"{case_name} {optimum} candidate {candidate_id}: copied and mapped", flush=True)

    with (DESTINATION / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Complete nonlinear runs and maps saved to {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()