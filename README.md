# Thypredict Pipeline

Thypredict is a deep learning (DL)-based, three-stage diagnostic pipeline for distinguishing NIFTP from a wide spectrum of thyroid lesions, including benign and malignant mimics using histopathology images. This pipeline has been tested on Ubuntu 22.04.4, CUDA compatible graphic card and RAM > 16GB.



### To get started: Select the region of interest from the scanned slide, crop it and save it as jpg/jpeg/png/tiff/ image. This image would be the input to the pipeline.
## Prerequisites

Before running the pipeline, ensure you have the following:

- A system with Python 3.10.12 installed
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


## Key Files

- **`MyAppInstaller_mcr.install`**: Installs the MATLAB Runtime. It's a large file (3.4GB) hosted externally and should be downloded from [https://apexbtic.icgeb.res.in/thypredict/MyAppInstaller_mcr.install](https://apexbtic.icgeb.res.in/thypredict/MyAppInstaller_mcr.install).
- **`MATLAB/`**: Blank directory where the MATLAB Runtime will be installed.
- **`input-image/`**: Directory to place input images for the pipeline.
- **`thypredict.py`**: Main file for stages I and II of the pipeline.
- **`utils.py`**: Contains functions used in `thypredict.py`.
- **`matlab-run.py`**: Executes the MATLAB Runtime for stage III.
- **`models/`**: Contains model files (~500MB). Due to size constraints, it is hosted separately and should be downloaded from [https://apexbtic.icgeb.res.in/thypredict/model/](https://apexbtic.icgeb.res.in/thypredict/model/).

## Installation and Running the Pipeline
1. **To install and set up the ThyPredict pipeline, follow these steps:**

    Clone the repository to your local machine:

```bash
git clone https://github.com/tbgicgeb/ThyPredict.git
```

2. **To install all required packages, run:**
```bash
pip install -r requirements.txt
```

3. **Navigate to the ThyPredict Directory**

   Ensure you are inside the `ThyPredict` directory before running any commands:

   ```bash
   cd ThyPredict
   ```

4. **Prepare the Input Images**

   Create the `input-image/` directory if it doesn't exist, and place your input images inside it:

   ```bash
   mkdir -p input-image
   cp /path/to/YOUR_INPUT_IMG input-image/
   ```

5. **Prepare the Models' Directory**

The `models/` directory contains essential files for the pipeline and is approximately 500MB. You can download it from [this link](https://apexbtic.icgeb.res.in/thypredict/model/) and place it in the root of your project directory.

   Create the `models/` directory and copy all required model files into it:

   ```bash
   mkdir -p models
   ```
   ***STAGE-I***
   ```
   cp /path/to/Stage-I.h5 models/
   ```
   ***STAGE-II***
   ```
   cp /path/to/Stage-II.h5  models/
   ```
   ***Preprocessing STAGE-II***
   ```
   cp /path/to/preprocessing-roi.keras models/
   ```
     
7. **Download and Run the Installer:**

   Download the `MyAppInstaller_mcr.install` file from [this link](https://apexbtic.icgeb.res.in/thypredict/MyAppInstaller_mcr.install) (hosted on our server due to its large size) and execute it using the following command:
   
   ```bash
   mkdir MATLAB
   ./MyAppInstaller_mcr.install
   
   *Note: It will take some time to open the GUI
   ```
   
A GUI will open, prompting you to provide two destination paths, (1) The Application and (2) MATLAB Runtime. You must provide the complete path to a directory named `MATLAB`.

7. **Run the Pipeline**

   Execute the `thypredict.py` script to start the initial processing:

   ```bash
   python thypredict.py image_name.jpg
   *Note: Do not enter the full path of the image. Only provide the image name that is inside the input-image directory
   ```

### Directory Structure

   After installation, your directory structure should resemble:

  ```
/path/to/Thypredict/
├── MATLAB/
│   ├── application/         # Contains application-specific files needed for the MATLAB runtime
│   └── R2023b/              # Specific version of the MATLAB runtime being used
├── input-image/             # Directory where input images are stored for processing
├── thypredict.py            # Main script for running the prediction pipeline
├── utils.py                 # Utility functions to support the main script
├── matlab.py                # Python script to interface with MATLAB runtime
└── models/                  # Directory containing machine learning model files (linked separately)
    ├── Stage-I.h5/          
    ├── Stage-II.h5/        
    └── preprocessing-roi.keras/  

   ```

## Handling Large Files

### MATLAB Runtime Installer

Due to GitHub's file size limits, the `MyAppInstaller_mcr.install` file is not stored directly in this repository. Instead, it is hosted on an external server. Please download it using the provided link.

## Contributions

Contributions are welcome! Please fork the repository and create a pull request for any improvements or fixes.

## License

This project is licensed under the [MIT License](LICENSE).


## If you have any questions, bug reports, or suggestions, please e-mail
Dr. Dinesh Gupta (dinesh@icgeb.res.in), [TRANSLATIONAL BIOINFORMATICS GROUP](https://bioinfo.icgeb.res.in/bioinfo/), [International Centre for Genetic Engineering and Biotechnology](https://www.icgeb.org/), New Delhi, India. 

