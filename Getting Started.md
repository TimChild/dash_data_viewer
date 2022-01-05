# Getting started with dash_data_viewer

1. Download Git Repo
    1. Clone repo from [GitHub](https://github.com/TimChild/dash_data_viewer) (easist to use [GitHub Desktop](https://desktop.github.com/))
    2. Make a new branch for yourself
2. Setup `dat_analysis` package following the "Getting Started" instructions at [dat_analysis](https://github.com/TimChild/dat_analysis)
3. With the same Conda environment install `dash_data_viewer`
    1. Activate the environment using `conda activate <name>`
    2. Install `dash_data_viewer` using `pip install <path to dash_data_viewer repo>`
        1. To install in an editable mode add the `-e` flag
4. Basic usage:
   1. From the conda terminal:
   ```bash
   cd <path to dash_data_viewer>/src/dash_data_viewer
   python dash2run.py  # Will run in debug mode
   python dash2run.py -r  # Will run in remote mode
    ```
