import subprocess
import glob
import os

# Define paths
matlab_path = "MATLAB/R2023b"
stageII_out = "_in_for_stage-III" #This is the input for stage-III
image_pattern = os.path.join(stageII_out, "*.tiff")

# Get a list of all image files
image_files = glob.glob(image_pattern)

# Loop through each image file and call the shell script
for image_file in image_files:
    try:

        # Print the image name with the full path
        print(f"Processing image: {image_file}")

        # Construct the command
        command = f"./MATLAB/application/run_classifyImageROI.sh {matlab_path} {image_file}"

        # Execute the command
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Print the result
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while processing {image_file}")
        print(e.stderr.decode())
