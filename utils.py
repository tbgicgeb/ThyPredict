import os
from tabulate import tabulate 

# Function to calculate and print image percentages
def calculate_image_percentages(save_dir):
    total_images = 0
    image_counts = {}

    # Calculate total images and count per folder
    for class_name in os.listdir(save_dir):
        class_dir = os.path.join(save_dir, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))]
            image_count = len(image_files)
            image_counts[class_name] = image_count
            total_images += image_count

    # Prepare table data
    table_data = []
    if total_images > 0:
        for class_name, count in image_counts.items():
            percentage = (count / total_images) * 100
            # Highlight the class if no images are found
            highlight = '\033[91m' if count == 0 else '\033[92m'  # Red for 0%, Green for others
            table_data.append([f"{highlight}{class_name}\033[0m", f"{percentage:.2f}%"])
    else:
        print("No images found in any class folder in 'main-crop'.")
        return

    # Display table
    headers = ["Class", "Percentage"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))



# New function to calculate and print percentages for PTC_Like and Non-PTC_like
def calculate_ptc_class_percentages(predict2):
    total_images = 0
    image_counts = {"Non-PTC-like_nuclear_fea": 0, "PTC-like_nuclear_fea": 0}

    # Calculate total images and count for PTC_Like and Non-PTC_like
    for class_name in image_counts.keys():
        class_dir = os.path.join(predict2, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))]
            image_count = len(image_files)
            image_counts[class_name] = image_count
            total_images += image_count

    # Prepare table data
    table_data = []
    if total_images > 0:
        for class_name, count in image_counts.items():
            percentage = (count / total_images) * 100
            table_data.append([class_name, f"{percentage:.2f}%"])
    else:
        print("No images found in PTC_Like or Non-PTC_like folders.")
        return

    # Display table
    headers = ["Class", "Percentage"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))




# New function to calculate and print percentages for PTC_Like and Non-PTC_like
def calculate_class_2n_3e_4i_percentages(final_result_dir):
    total_images = 0
    image_counts = {"class2n": 0, "class3e": 0, "class4i": 0}

    # Calculate total images and count for PTC_Like and Non-PTC_like
    for class_name in image_counts.keys():
        class_dir = os.path.join(final_result_dir, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))]
            image_count = len(image_files)
            image_counts[class_name] = image_count
            total_images += image_count

    # Prepare table data
    table_data = []
    if total_images > 0:
        for class_name, count in image_counts.items():
            percentage = (count / total_images) * 100
            table_data.append([class_name, f"{percentage:.2f}%"])
    else:
        print("No images found in class2n, class3e, class4i  folders.")
        return

    # Display table
    headers = ["Class", "Percentage"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
