# Author: Ben Thompson
# Python 3.13.1
# Version 1.1

import cv2
import os
import numpy as np
import sys
import json

print("OpenCV Bulk Face Crop has started")

def loadJSON(config_file):
    """
    Loads a JSON configuration file.
    If the file doesn't exist, it returns an empty dictionary.
    """
    # Checks if the JSON file exists
    if os.path.exists(config_file):
        try:
            # Reads the JSON file and assigns it to object f
            with open(config_file, 'r') as f:
                # Loads the jsoncontents into a Python dictionary
                return json.load(f)
        except json.JSONDecodeError as e:
            # Provide detailed error information
            print("Error: Failed to decode JSON in file '{filepath}'.")
            print("Details: {e.msg} at line {e.lineno}, column {e.colno} (character {e.pos}).")
            print("Hint: Check for syntax errors in the JSON file, such as missing quotes or trailing commas.")
        except Exception as e:
            # Handle any other unexpected exceptions
            print("Error: An unexpected error occurred while loading '{filepath}': {e}")
    else:
        print(f"Config file {config_file} not found! Using default settings.")
        return {}

def setDictionaries(conf):
    # Gets the values inside of face_detect from the config and sets it to an empty dictionary if not found
    face_detect = conf.get("face_detection", [])
    # Gets the values inside of output_dimensionst from the config and sets it to an empty dictionary if not found
    output_dims = conf.get("output_dimensions", [])
    # Gets the values inside of output_image_details from the config and sets it to an empty dictionary if not found
    out_img_deets = conf.get("output_image_details", [])
    # Gets the values inside of paths from the config and sets it to an empty dictionary if not found
    path_dict = conf.get("paths", [])
    # Gets the values inside of the terminal from the config and sets it to an empty dictionary if not found
    terminal_vals = conf.get("terminal", [])

    return face_detect, output_dims, out_img_deets, path_dict, terminal_vals
    
    

def getCascadePath():
    """
    Returns the correct path to the Haar Cascade file.
    This works for both normal Python execution and when bundled by PyInstaller.
    """
    if getattr(sys, 'frozen', False):
        # If running as an executable, use the temporary directory created by PyInstaller
        return os.path.join(sys._MEIPASS, 'cv2', 'data', 'haarcascade_frontalface_default.xml')
    else:
        # If running as a script, use the regular file path
        return cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def adjustBoundingBoxToAspectRatio(x, y, w, h, target_aspect_ratio, image_width, image_height):
    """
    Adjusts the bounding box to match the aspect ratio of the output width and height,
    expanding the box if needed but keeping it within image bounds.
    """
    # Calculates the current aspect ratio of the bounding box
    current_aspect_ratio = w / h

    if current_aspect_ratio < target_aspect_ratio:
        # If the current aspect ratio is less than the target (box is too tall)
        # Calculates the new width to match the target aspect ratio
        new_width = int(h * target_aspect_ratio)
        # Calculate how much the width needs to expand on each side
        diff = (new_width - w) // 2
        # Adjust the x-coordinate to expand the box symmetrically
        # Ensure it doesn't move outside the left image boundary
        x = max(0, x - diff)
        # Update the width, ensuring it doesn't exceed the image's right boundary
        w = min(new_width, image_width - x)
    else:
        # If the current aspect ratio is greater than the target (box is too wide)
        # Calculates the new height to match the target aspect ratio
        new_height = int(w / target_aspect_ratio)
        # Calculate how much the height needs to expand on each side
        diff = (new_height - h) // 2
        # Adjust the y-coordinate to expand the box symmetrically
        # Ensure it doesn't move outside the top image boundary
        y = max(0, y - diff)
        # Update the height, ensuring it doesn't exceed the image's bottom boundary
        h = min(new_height, image_height - y)

    # Returns the adjusted bounding box dimensions
    return x, y, w, h

def cropAndResize(passed_output_aspect_ratio, passed_output_px_width, passed_output_px_height, image, x, y, w, h):
    """
    Crop the image and resize it to the output size without adding unnecessary padding.
    Expands the crop to fill the output size as much as possible.
    """
    # Gets the original height and width of the input image that is in the array
    image_height, image_width = image.shape[:2]

    # Adjust the bounding box to match the target aspect ratio, keeping it within the image boundaries
    x, y, w, h = adjustBoundingBoxToAspectRatio(x, y, w, h, passed_output_aspect_ratio, image_width, image_height)

    # Crops the image according to the adjusted bounding box (x, y, w, h)
    cropped = image[y:y+h, x:x+w]

    # Resize the cropped image to match the target output size (width, height)
    resized = cv2.resize(cropped, (passed_output_px_width, passed_output_px_height))

    # Returns the resized image
    return resized

# Set the face detection accuracy values
def getFaceDetectAccuracy(face_dict):
    face_detect_acc = face_dict.get("accuracy", None)

    if face_detect_acc is None:
        print("Check the JSON config, the accuracy value inside of face_detection couldn't be found, default value of 1.1 has been set instead")
        face_detect_acc = 1.1
    elif face_detect_acc <= 1:
        print("Check the JSON config, the accuracy value inside of face_detection must be greater than 1, default value of 1.1 has been set instead")
        face_detect_acc = 1.1

    return face_detect_acc

def getMinOverlapNum(face_dict):
    min_overlay_val = face_dict.get("min_overlap_num", None)

    if min_overlay_val is None:
        print("Check the JSON config, the min_overlap_num value inside of face_detection couldn't be found, default value of 6 has been set instead")
        min_overlay_val = 6
    elif min_overlay_val < 0:
        print("Check the JSON config, the min_overlap_num inside of face_detection cannot be less than 0, default value of 6 has been set instead")
        min_overlay_val = 6

    return min_overlay_val

def getMinFaceWidth(face_dict):
    face_min_w_px = face_dict.get("min_face_width", None)

    if face_min_w_px is None:
        print("Check the JSON config, the min_face_width value inside of face_detection couldn't be found, default value of 30 has been set instead")
        face_min_w_px = 30
    elif face_min_w_px < 1:
        print("Check the JSON config, the min_face_width inside of face_detection cannot be less than 1, default value of 30 has been set instead")
        face_min_w_px = 30

    return face_min_w_px

def getMinFaceHeight(face_dict):
    face_min_h_px = face_dict.get("min_face_height", None)

    if face_min_h_px is None:
        print("Check the JSON config, the min_face_height value inside of face_detection couldn't be found, default value of 30 has been set instead")
        face_min_h_px = 30
    elif face_min_h_px < 1:
        print("Check the JSON config, the min_face_height inside of face_detection cannot be less than 1, default value of 30 has been set instead")
        face_min_h_px = 30

    return face_min_h_px

def getFaceMargin(face_dict):
    face_margin = face_dict.get("margin", None)

    if face_margin is None:
        print("Check the JSON config, the margin value inside of face_detection couldn't be found, default value of 0.4 has been set instead")
        face_margin = 0.4
    elif face_margin < 0:
        print("Check the JSON config, the margin inside of face_detection cannot be less than 0, default value of 0.4 has been set instead")
        face_margin = 0.4

    return face_margin

def getOutputWidth(dimensions_dict):
    width_px = dimensions_dict.get("width", None)

    if width_px is None:
        print("Check the JSON config, the width value inside of output_dimensions couldn't be found, default value of 320 has been set instead")
        width_px = 320
    elif width_px < 1:
        print("Check the JSON config, the width inside of output_dimensions cannot be less than 1, default value of 320 has been set instead")
        width_px = 320

    return width_px

def getOutputHeight(dimensions_dict):
    height_px = dimensions_dict.get("height", None)

    if height_px is None:
        print("Check the JSON config, the height value inside of output_dimensions couldn't be found, default value of 560 has been set instead")
        height_px = 560
    elif height_px < 1:
        print("Check the JSON config, the width inside of output_dimensions cannot be less than 1, default value of 560 has been set instead")
        height_px = 560

    return height_px

def getOutputFileType(img_details_dict):
    img_file_type = img_details_dict.get("file_type", None)

    if img_file_type is None:
        print("Check the JSON config, the file_type value inside of output_image_details couldn't be found, default value of jpg has been set instead")
        img_file_type = "jpg"
    elif img_file_type != "jpg" and img_file_type != "png" and img_file_type != "bmp" and img_file_type != "webp":
        print("Check the JSON config, the file_type inside of output_image_details is not jpg, png, bmp or webp, default value of jpg has been set instead")
        img_file_type = "jpg"

    return img_file_type

def getImageQuality(img_details_dict):
    img_compression = img_details_dict.get("image_quality", None)

    if img_compression is None:
        print("Check the JSON config, the image_quality value inside of output_image_details couldn't be found, default value of 100 has been set instead")
        img_compression = 100
    elif img_compression < 0 or img_compression > 100:
        print("Check the JSON config, the image_quality inside of output_image_details is out of bounds the value must be between 0 to 100, default value of 100 has been set instead")
        img_compression = 100

    return img_compression

def getCreateOutputDir(dir_dict):
    create_out_dir_bool = dir_dict.get("create_output_dir", None)

    if create_out_dir_bool is None:
        print("Check the JSON config, the create_output_dir value inside of paths couldn't be found, default value of False has been set instead")
        create_out_dir_bool = False
    elif create_out_dir_bool != "true" and create_out_dir_bool != "True" and create_out_dir_bool != "false" and create_out_dir_bool != "False":
        print("Check the JSON config, the create_output_dir value inside of paths is invalid, default value of False has been set instead")
        create_out_dir_bool = False
    elif create_out_dir_bool == "true" or create_out_dir_bool == "True":
        create_out_dir_bool = True
    elif create_out_dir_bool == "false" or create_out_dir_bool == "False":
        create_out_dir_bool = False

    return create_out_dir_bool

def getAllowSameInputOutputDir(dir_dict):
    input_ouput_dir_same_bool = dir_dict.get("allow_same_input_output_dir", None)

    if input_ouput_dir_same_bool is None:
        print("Check the JSON config, the allow_same_input_output_dir value inside of paths couldn't be found, default value of False has been set instead")
        input_ouput_dir_same_bool = False
    elif input_ouput_dir_same_bool != "true" and input_ouput_dir_same_bool != "True" and input_ouput_dir_same_bool != "false" and input_ouput_dir_same_bool != "False":
        print("Check the JSON config, the allow_same_input_output_dir inside of paths is invalid, default value of False has been set instead")
        input_ouput_dir_same_bool = False
    elif input_ouput_dir_same_bool == "true" or input_ouput_dir_same_bool == "True":
        input_ouput_dir_same_bool = True
    elif input_ouput_dir_same_bool == "false" or input_ouput_dir_same_bool == "False":
        input_ouput_dir_same_bool = False

    return input_ouput_dir_same_bool
        

def getInputFolder(dir_dict):
    input_dir = dir_dict.get("input_folder", None)

    if input_dir is None:
        sys.exit("Check the JSON config, the input_folder value inside of paths couldn't be found, please resolve the issue")
    elif input_dir == "":
        sys.exit("Check the JSON config, the input_folder value inside of paths couldn't be found, please resolve the issue")
    elif not os.path.isdir(input_dir):
        sys.exit("Check the input_folder value inside of paths for the JSON config file. The specified directory does not exist")

    return input_dir

def getOutputFolder(dir_dict):
    output_dir = dir_dict.get("output_folder", None)

    if output_dir is None:
        sys.exit("Check the JSON config, the output_folder value inside of paths couldn't be found, please resolve the issue")
    elif output_dir == "":
        sys.exit("Check the JSON config, the output_folder value inside of paths couldn't be found, please resolve the issue")
    elif not os.path.isdir(output_dir):
        # Calls the function that reads the create_output_dir value from the config file to decide if the output directoy that doesn't exist should be created
        create_output_dir = getCreateOutputDir(dir_dict)

        if create_output_dir:
            print("Warning! The specified output directory doesn't exists. The output directory will be created")
        else:
            sys.exit("Check the output_folder value inside of paths for the JSON config file. The specified directory does not exist")

    return output_dir

def getPrintProcessedImage(terminal_dict):
    print_processed_img_bool = terminal_dict.get("print_processed_image", None)

    if print_processed_img_bool is None:
        print("Check the JSON config, the print_processed_image value inside of terminal couldn't be found, default value of True has been set instead")
        print_processed_img_bool = True
    elif print_processed_img_bool != "true" and print_processed_img_bool != "True" and print_processed_img_bool != "false" and print_processed_img_bool != "False":
        print("Check the JSON config, the print_processed_image inside of terminal is invalid, default value of True has been set instead")
        print_processed_img_bool = True
    elif print_processed_img_bool == "true" or print_processed_img_bool == "True":
        print_processed_img_bool = True
    elif print_processed_img_bool == "false" or print_processed_img_bool == "False":
        print_processed_img_bool = False

    return print_processed_img_bool

def processImages(passed_face_detection, passed_output_dimensions, passed_output_image_details, passed_paths, passed_terminal):
    # Sets the variable used for the scale factor
    face_detect_accuracy = getFaceDetectAccuracy(passed_face_detection)
    # Sets the variable used for the nim neighbors parameter
    min_overlap_num = getMinOverlapNum(passed_face_detection)
    # Sets the variable that is used for the minimum width of the bounding box for a detection to be valid
    min_face_width = getMinFaceWidth(passed_face_detection)
    # Sets the variable that is used for the minimum height of the bounding box for a detection to be valid
    min_face_height = getMinFaceHeight(passed_face_detection)
    # Sets the variable that is used for the amount of margin % around the detected face
    margin = getFaceMargin(passed_face_detection)

    # Sets the variable that is used to set the width in px of the output images
    output_px_width = getOutputWidth(passed_output_dimensions)    # Sets the variable that is used to set the height in px of the output images
    output_px_height = getOutputHeight(passed_output_dimensions)

    # Sets the variable that is used for the file type of the output images
    output_file_type = getOutputFileType(passed_output_image_details)
    # Sets the variable that is used for the amount of compression on the output images
    image_quality = getImageQuality(passed_output_image_details)

    # Sets the variable that is used for the input directory for the images
    input_folder = getInputFolder(passed_paths)
    output_folder = getOutputFolder(passed_paths)

    # Checks if the input and output directories are the same
    if input_folder == output_folder:
        allow_same_dir = getAllowSameInputOutputDir(passed_paths)

        if allow_same_dir:
            print("Warning! The input and output directories are the same. The output images will overwrite the input images if the ouput file type matches the input file type of the image")
        else:
            sys.exit("Warning! The input and output directories are the same. The program has terminated to prevent the output images overwriting the input images of the same image file type")

    # Sets the variable that is used to determin if the file name of the images that has been processed should be printed.
    display_processed_img_name = getPrintProcessedImage(passed_terminal)

    # Calulates the aspect ratio that the output should be and stores it
    output_aspect_ratio = output_px_width / output_px_height
    output_size = (output_px_width, output_px_height)

    # Checks if the output directory exists and if it doesn't creates it
    os.makedirs(output_folder, exist_ok=True)

    # Loads the Haar Cascade face detection model
    face_cascade = cv2.CascadeClassifier(getCascadePath())

    # Creates a list of all the image names with the file extensions that or in the input directory
    input_img_list = os.listdir(input_folder)

    if len(input_img_list) > 0:
        # Loops through all files in the input folder
        for filename in input_img_list:
            # Process only image files with .jpg, .png, .bmp or .webp extensions
            if filename.endswith((".jpg", ".png", ".bmp", ".webp")):
                #Combines the input directory with the file name for the full directory location of the image
                image_path = os.path.join(input_folder, filename)
                #Loads the image into a NumPy multi-dimensional array containting the RGB values of each pixel
                image_rgb_array = cv2.imread(image_path)
                #Creates a grayscake 2D array of the image to be used later as Haar Cascades is more efficient in greyscale
                image_grey_array = cv2.cvtColor(image_rgb_array, cv2.COLOR_BGR2GRAY)
                #Gets the height and width values of the input image
                image_height, image_width = image_rgb_array.shape[:2]

                # Attempts to detect a face in the image and if it does it stores the horizontal top left value,
                # vertical top left value, box width and box height in a list
                faces = face_cascade.detectMultiScale(
                    image_grey_array, face_detect_accuracy, min_overlap_num, minSize=(min_face_width, min_face_height)
                )

                # Checks if the faces list is populated, if it isn't then no face was detected in the image
                if len(faces) > 0:
                    # Selects the first detected face in the list (x, y, width, height)
                    x, y, w, h = faces[0]

                    # Calcualtes the margin in px based on the larger dimension (width or height) and the resoultion of the face
                    px_margin_size = int(max(w, h) * margin)

                    # Add margin around the adjusted bounding box
                    x = max(0, x - px_margin_size)
                    y = max(0, y - px_margin_size)
                    w = min(w + 2 * px_margin_size, image_width - x)
                    h = min(h + 2 * px_margin_size, image_height - y)

                    # Crop and resize after bounding box adjustment
                    resized_face = cropAndResize(output_aspect_ratio, output_px_width, output_px_height, image_rgb_array, x, y, w, h)

                    # Seprates the filename from the file type so that the output can be standerdised
                    filename_no_file_type = os.path.splitext(filename)
                    #Combines the output directory and the image file name into a single full directory location
                    output_path = os.path.join(output_folder, filename_no_file_type[0] + "." + output_file_type)

                    # Save the processed image to the output folder
                    if output_file_type == 'jpg' or output_file_type == 'jpeg':
                        # JPEG format: Adjust quality (0 to 100)
                        cv2.imwrite(output_path, resized_face, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
                    elif output_file_type == 'png':
                        # PNG format: Adjust compression (0 to 9, where 0 is best quality and 9 is worst)
                        # Reverse the image_quality mapping since 0 is best for PNG
                        compression_level = 9 - (image_quality // 10)  # Adjust quality to compression level (0 = best, 9 = worst)
                        cv2.imwrite(output_path, resized_face, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
                    elif output_file_type == 'bmp':
                        # BMP format: No compression, as BMP does not support compression through cv2.imwrite
                        cv2.imwrite(output_path, resized_face)
                    elif output_file_type == 'webp':
                        # WebP format: Adjust quality (0 to 100)
                        cv2.imwrite(output_path, resized_face, [cv2.IMWRITE_WEBP_QUALITY, image_quality])
                    else:
                        print("Output file type is not supported!")

                    # Prints the name of the processed image if the value is set to True in the config
                    if display_processed_img_name:
                        print(f"Processed {filename}")
                else:
                    print(f"No face detected in {filename}")
    else:
        print("Warning! No images of supported file type were found in the input directory, check the input_folder value inside of paths that is contained within the config JSON file")

def getTerminalAutoClose(terminal_dict):
    auto_close_bool = terminal_dict.get("terminal_auto_close", None)

    if auto_close_bool is None:
        print("Check the JSON config, the terminal_auto_close value inside of terminal couldn't be found, default value of False has been set instead")
        auto_close_bool = False
    elif auto_close_bool != "true" and auto_close_bool != "True" and auto_close_bool != "false" and auto_close_bool != "False":
        print("Check the JSON config, the terminal_auto_close inside of terminal is invalid, default value of False has been set instead")
        auto_close_bool = False
    elif auto_close_bool == "true" or auto_close_bool == "True":
        auto_close_bool = True
    elif auto_close_bool == "false" or auto_close_bool == "False":
        auto_close_bool = False

    return auto_close_bool


# Calls the function that loads the external JSON config file
config = loadJSON("config.json");
# Sets the dictionaries for the config values
face_detection, output_dimensions, output_image_details, paths, terminal = setDictionaries(config);

# Process the images
processImages(face_detection, output_dimensions, output_image_details, paths, terminal)

# Checks if the terminal should be close automatically after the program finishes, or should wait for a key input to close the terminal
if getTerminalAutoClose(terminal):
    print("Done")
else:
    print("Done")
    input("Press any key to close the terminal")

