# 15.285-ContestedProject

## Load parquet files from local OneDrive

This project includes a starter script to read parquet files from:

`/Users/mariaangellobon/Library/CloudStorage/OneDrive-SharedLibraries-MassachusettsInstituteofTechnology/[MIT] Basketball Officiating - miami_heat_2025`

### Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run:
   `python load_parquet_from_onedrive.py`

Optional arguments:

- `--folder`: override the default folder path.
- `--sample-rows`: number of preview rows from the first file (default: 5).

## Notes

Points to think about:
- be careful about overclaiming with regards to the measure's ability to incorporate defender intent; the data likely provides imputed intent at best
- when presenting a new metric, include intuitive scale context (min, max, central tendency)
