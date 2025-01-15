# Thypridict Pipeline

Thypridict is a pipeline for analyzing and predicting thyroid image. This repository contains the essential scripts and instructions to get started.

## Prerequisites

Before running the pipeline, ensure you have the following:

- A system with Python installed
- MATLAB Runtime environment

### Installation Steps

1. **Download and Run the Installer:**

   Download the `MyAppinstaller_mrc.install` file from [this link](https://apexbtic.icgeb.res.in/thypredict/MyAppInstaller_mcr.install) (hosted on our server due to its large size) and execute it using the following command:
   
   ```bash
   $ ./MyAppinstaller_mrc.install
   ```
   
   A GUI will open, prompting you to provide the application and MATLAB Runtime destination paths. You must provide the complete path to a directory named `MATLAB`. This directory will house the installed MATLAB Runtime environment.

2. **Directory Structure:**

   After installation, your directory structure should resemble:

  ```
   /path/to/Thypredict/
   ├── MATLAB/
   │   ├── application/
   │   └── R2023b/
   ├── input-image/
   ├── thipredict.py
   ├── utils.py
   ├── matlab.py
   └── models/ (linked separately)
   |   ├── Stage-I.h5/
   │   |── Stage-II.h5/
   |   |__ Stage-II-roi.h5/
   ```

## Key Files

- **`MyAppinstaller_mrc.install`**: Installs the MATLAB Runtime. It's a large file (3.4GB) hosted externally and should be downloded from [https://apexbtic.icgeb.res.in/thypredict/MyAppInstaller_mcr.install](https://apexbtic.icgeb.res.in/thypredict/MyAppInstaller_mcr.install).
- **`MATLAB/`**: Blank directory where the MATLAB Runtime will be installed.
- **`input-image/`**: Directory to place input images for the pipeline.
- **`thipredict.py`**: Main file for stages I and II of the pipeline.
- **`utils.py`**: Contains functions used in `thipredict.py`.
- **`matlab-run.py`**: Executes the MATLAB Runtime for stage III.
- **`models/`**: Contains model files (~500MB). Due to size constraints, it is hosted separately and should be downloaded from [https://apexbtic.icgeb.res.in/thypredict/model/](https://apexbtic.icgeb.res.in/thypredict/model/).

## Handling Large Files

### MATLAB Runtime Installer

Due to GitHub's file size limits, the `MyAppinstaller_mrc.install` file is not stored directly in this repository. Instead, it is hosted on an external server. Please download it using the provided link.

### Models Directory

The `models/` directory contains essential files for the pipeline and is approximately 500MB. This directory is also hosted externally. You can download it from [this link](#) and place it in the root of your project directory.

## Running the Pipeline

1. Place your input images in the `input-image/` directory.
2. Run the `thipredict.py` script for initial processing:
   
   ```bash
   $ python thipredict.py
   ```

## Contributions

Contributions are welcome! Please fork the repository and create a pull request for any improvements or fixes.

## License

This project is licensed under the [MIT License](LICENSE).
