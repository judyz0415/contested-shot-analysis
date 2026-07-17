from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ONEDRIVE_PATH = Path(
    "/Users/mariaangellobon/Library/CloudStorage/"
    "OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/"
    "[MIT] Basketball Officiating - miami_heat_2025"
)


def list_parquet_files(folder: Path) -> list[Path]:
    """Return sorted parquet files from a folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder}")
    return sorted(folder.glob("*.parquet"))


def load_first_file_sample(parquet_files: list[Path], rows: int) -> pd.DataFrame:
    """Load a small preview from the first parquet file."""
    if not parquet_files:
        raise ValueError("No parquet files found in the target folder.")
    return pd.read_parquet(parquet_files[0]).head(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List and preview parquet files from local OneDrive folder."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_ONEDRIVE_PATH,
        help="Folder containing parquet files.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of rows to print from the first parquet file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Write the sample rows to this CSV path (all columns). "
            "Opens cleanly in Excel, Numbers, or the editor."
        ),
    )
    args = parser.parse_args()

    parquet_files = list_parquet_files(args.folder)
    print(f"Folder: {args.folder}")
    print(f"Found {len(parquet_files)} parquet files.")
    for file_path in parquet_files:
        print(f"- {file_path.name}")

    sample_df = load_first_file_sample(parquet_files, args.sample_rows)
    print("\nSample rows from first parquet file:")
    print(sample_df)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"\nSaved sample CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
