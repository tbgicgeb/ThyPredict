# Thypredict Pipeline

Thypridict is a pipeline for analyzing and predicting thyroid image. This repository contains the essential scripts and instructions to get started.

## Prerequisites

Before running the pipeline, ensure you have the following:

- A system with Python installed
  - The following Python packages are required:
  - `image_slicer==2.1.1`
  - `keras==2.15.0`
  - `matplotlib==3.6.0`
  - `numpy==1.23.0`
  - `opencv_python==4.10.0.84`
  - `Pillow==7.2.0`
  - `Pillow==11.1.0`
  - `protobuf==4.24.4`
  - `scikit_learn==1.6.1`
  - `tabulate==0.9.0`
  - `tensorflow==2.15.0`

To install all required packages, run:

```bash
pip install -r requirements.txt
```
- MATLAB Runtime environment

### Installation Steps

1. **To install and set up the ThyPredict pipeline, follow these steps:**

    Clone the repository to your local machine:

```bash
$ git clone https://github.com/tbgicgeb/ThyPredict.git
```



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
1. **Navigate to the ThyPredict Directory**

   Ensure you are inside the `ThyPredict` directory before running any commands:

   ```bash
   $ cd ThyPredict
   ```

2. **Prepare the Input Images**

   Create the `input-image/` directory if it doesn't exist, and place your input images inside it:

   ```bash
   $ mkdir -p input-image
   $ cp /path/to/YOUR_INPUT_IMG input-image/
   ```

3. **Prepare the Models**

   Create the `models/` directory and copy all required model files into it:

   ```bash
   $ mkdir -p models
   ```
   ***STAGE-I***
   ```
   $ cp /path/to/Stage-I.h5 models/
   ```
   ***STAGE-II***
   ```
   $ cp /path/to/Stage-II.h5  models/
   ```
   ***Preprocessing STAGE-II***
   ```
   $ cp /path/to/preprocessing-roi.keras models/
   ```
     
4. **Download and Run the Installer:**

   Download the `MyAppinstaller_mrc.install` file from [this link](https://apexbtic.icgeb.res.in/thypredict/MyAppInstaller_mcr.install) (hosted on our server due to its large size) and execute it using the following command:
   
   ```bash
   $ mkdir MATLAB
   $ ./MyAppinstaller_mrc.install
   ```
   
A GUI will open, prompting you to provide two destination paths, (1) The Application and (2) MATLAB Runtime . You must provide the complete path to a directory named `MATLAB`. This directory will house the installed MATLAB Runtime environment.

4. **Run the Pipeline**

   Execute the `thipredict.py` script to start the initial processing:

   ```bash
   $ python thipredict.py
   ```


## Contributions

Contributions are welcome! Please fork the repository and create a pull request for any improvements or fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Computational Cost
- approximate time  90 min for 100 samples, ~4000 features and ~40000 interactions with 5000 decision trees.
- 256 GB RAM and Intel® Xeon(R) Silver 4114 CPU @ 2.20GHz × 40 processors.

## If you have any questions, bug reports, or suggestions, please e-mail
Dr. Dinesh Gupta (dinesh@icgeb.res.in), TRANSLATIONAL BIOINFORMATICS GROUP, International Centre for Genetic Engineering and Biotechnology, New Delhi, India. 

