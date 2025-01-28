import math
import os
import subprocess
import numpy as np
import json

# import generated and filtered data to use here
from generate_data import lab_grid, lab_points, filtered_points, filtered_points_nparray

# here we take the data and evaluate it, returning the differences in deltas


folder_path_input = "Evaluate_RGB_Profiles/Input/"
folder_path_output = "Evaluate_RGB_Profiles/Output/"
profile = "Evaluate_RGB_Profiles/Input/CS_sep_AdobeRGB_2_ISOcoatedv2-39L_TAC330_V6.icc"


def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def is_all_floats(data):

    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        return False

    for row in data:
        for element in row:
            if not isinstance(element, float):
                return False
    return True


def format_data(input_data):
    # Check if input_data is a list of lists with all floats
    if is_all_floats(input_data):
        return input_data

    # Otherwise, format the string data into a list of lists
    input_data = input_data.replace("\t", " ").splitlines()
    input_data = [row.split() for row in input_data]

    # Convert all elements to floats
    input_data = [[float(item) for item in row] for row in input_data]
    return input_data


def get_all_deltas(vectors):

    # arrays of delta e's, l's, c's and h's
    delta_E_arr = []
    delta_L_arr = []
    delta_C_arr = []
    delta_H_arr = []

    # container for one C
    C = []

    for vector in vectors:
        p1, p2 = vector
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)

        deltaE = float(np.linalg.norm(p1 - p2))
        delta_E_arr.append(deltaE)

        deltaL = float(p1[0] - p2[0])
        delta_L_arr.append(float(p1[0] - p2[0]))

        # Compute C for both points (distance to origin using 2nd and 3rd coordinates)
        C1 = float(np.linalg.norm(p1[1:]))
        C2 = float(np.linalg.norm(p2[1:]))
        C.append((C1, C2))  # Store both C values as a tuple

        # Compute absolute difference between C values
        # is it ok to use abs here?
        deltaC = abs(C1 - C2)
        delta_C_arr.append(deltaC)

        deltaH = float(math.sqrt((deltaE**2) - (deltaL**2) - (deltaC**2)))
        delta_H_arr.append(deltaH)

    return {
        "deltaE": delta_E_arr,
        "deltaL": delta_L_arr,
        "deltaC": delta_C_arr,
        "deltaH": delta_H_arr,
    }


def compare_deltas(deltas1, deltas2):
    comparison = {}
    for key in deltas1:
        if key in deltas2:
            diff = f"{key}diff"
            comparison[diff] = [
                -1 * (v1 - v2) for v1, v2 in zip(deltas1[key], deltas2[key])
            ]
        else:
            # this errorcatch should not be useful in this scenario but just in case
            raise ValueError(f"Key '{key}' not found in deltas2.")

    return comparison


def extract_neighboring_slices(flat_grid: np.ndarray, step=1):
    """
    Extract neighboring slices from a 2D array of LAB points.

    Parameters:
        flat_grid (np.ndarray): A 2D NumPy array of LAB points with shape (n, 3).
        step (int): Step size to extract neighboring slices.

    Returns:
        list of tuple: A list of tuples, where each tuple contains two neighboring slices.
    """
    slices = []
    n_points = len(flat_grid)

    # Ensure enough points for slicing
    for i in range(0, n_points - step, step):
        slice1 = flat_grid[i : i + step]
        slice2 = flat_grid[i + step : i + 2 * step]
        slices.append((slice1, slice2))

    return slices


original_shape = filtered_points_nparray.shape


# this is a helper function to take the slices in our format and make them a tab delimited string
def convert_slices_to_tab_delimited_individual(slice_pairs):
    """
    Converts each pair of neighboring slices into two separate tab-delimited strings
    matching the exact format of the provided file.

    Parameters:
        slice_pairs (list of tuple): List of tuples, where each tuple contains two slices (2D arrays).

    Returns:
        list of tuple: A list of tuples, where each tuple contains two formatted strings
                       (one for each slice in the pair).
    """
    formatted_strings = []

    for slice1, slice2 in slice_pairs:
        # Flatten both slices into a 2D array with shape (N, 3)
        slice1_flat = slice1.reshape(-1, slice1.shape[-1])
        slice2_flat = slice2.reshape(-1, slice2.shape[-1])

        # Convert slice1 to a tab-delimited string
        slice1_formatted = "\n".join(
            "\t".join(f"{value:g}" for value in row) for row in slice1_flat
        )

        # Convert slice2 to a tab-delimited string
        slice2_formatted = "\n".join(
            "\t".join(f"{value:g}" for value in row) for row in slice2_flat
        )

        # Append the result to the list as a tuple
        formatted_strings.append((slice1_formatted, slice2_formatted))

    return formatted_strings


def clear_temporary_files(temp_folder="temporary_files"):
    """
    Clears all files in the specified temporary folder.

    Parameters:
        temp_folder (str): The folder to clear.
    """
    if os.path.exists(temp_folder):
        for file in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty directories
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        os.makedirs(temp_folder)  # Ensure the folder exists


def write_temp_files(lab_points, temp_folder="temporary_files"):
    """
    Writes neighboring slices as temporary files into a specified folder.

    Parameters:
        lab_points (np.ndarray): The flattened LAB grid.
        original_shape (tuple): The original 3D shape of the LAB grid.
        temp_folder (str): The folder to save the temporary files.

    Returns:
        list of str: Paths to the created temporary files.
    """

    clear_temporary_files(temp_folder)
    # Ensure the folder exists
    os.makedirs(temp_folder, exist_ok=True)

    # Extract neighboring slices
    neighboring_slices = extract_neighboring_slices(lab_points)
    print("Number of neighboring slices:", len(neighboring_slices))

    # Generate tab-delimited strings for the slices
    tab_delimited_strings = convert_slices_to_tab_delimited_individual(
        neighboring_slices
    )

    # Store the paths of the created files
    file_paths = []

    # Write each slice to a temporary file
    for i, (slice1, slice2) in enumerate(tab_delimited_strings):
        slice1_path = os.path.join(temp_folder, f"slice1_{i}.txt")
        slice2_path = os.path.join(temp_folder, f"slice2_{i}.txt")

        with open(slice1_path, "w", newline="\n") as f1:
            f1.write(slice1)
        with open(slice2_path, "w", newline="\n") as f2:
            f2.write(slice2)

        file_paths.extend([slice1_path, slice2_path])

    print(f"Temporary files written to {temp_folder}")
    return file_paths


# now the original and converted slices are stored in these two
# variables. they are tab delimited so
# original_slices, converted_slices = convert(lab_points, profile, original_shape)

# print(original_slices[:2])
# print(converted_slices[:2])


# this function works at last!!!
def take_files_from_folder_and_convert(Input_Lab1, Input_Lab2, profile):

    print("start")

    # Extract neighboring slices
    neighboring_slices = extract_neighboring_slices(lab_points, original_shape)
    print("Number of neighboring slices:", len(neighboring_slices))

    # Generate tab-delimited strings for the slices
    tab_delimited_strings = convert_slices_to_tab_delimited_individual(
        neighboring_slices
    )

    inputfile = profile
    print(inputfile)
    outputfile = folder_path_output + os.path.splitext(profile)[0] + ".txt"
    dstP = os.path.splitext(profile)[0] + ".dstP"
    print(dstP)
    inpP = os.path.splitext(profile)[0] + ".inpP"
    print(inpP)

    subprocess.run(["scripts/icc2tags", inputfile, "dstP"])
    subprocess.run(["scripts/icc2tags", inputfile, "inpP"])

    with open(Input_Lab1, "r") as file:
        Input_Lab1 = file.read()

    with open(Input_Lab2, "r") as file:
        Input_Lab2 = file.read()

    inputRGB1 = subprocess.run(
        [
            "scripts/transiccgmg.exe",
            "-o",
            inpP,
            "-i",
            "*Lab",
            "-t",
            "3",
            "-c",
            "0",
            "-n",
        ],
        input=Input_Lab1,
        text=True,
        capture_output=True,
    )

    inputRGB2 = subprocess.run(
        [
            "scripts/transiccgmg.exe",
            "-o",
            inpP,
            "-i",
            "*Lab",
            "-t",
            "3",
            "-c",
            "0",
            "-n",
        ],
        input=Input_Lab2,
        text=True,
        capture_output=True,
    )

    outputCLR1 = subprocess.run(
        ["scripts/transiccgmg.exe", "-l", inputfile, "-c", "0", "-n"],
        input=str(inputRGB1.stdout),
        text=True,
        capture_output=True,
    )

    outputCLR2 = subprocess.run(
        ["scripts/transiccgmg.exe", "-l", inputfile, "-c", "0", "-n"],
        input=str(inputRGB2.stdout),
        text=True,
        capture_output=True,
    )

    Output_Lab_1 = subprocess.run(
        [
            "scripts/transiccgmg.exe",
            "-i",
            dstP,
            "-o",
            "*Lab",
            "-t",
            "3",
            "-c",
            "0",
            "-n",
        ],
        input=outputCLR1.stdout,
        text=True,
        capture_output=True,
    )

    Output_Lab_2 = subprocess.run(
        [
            "scripts/transiccgmg.exe",
            "-i",
            dstP,
            "-o",
            "*Lab",
            "-t",
            "3",
            "-c",
            "0",
            "-n",
        ],
        input=outputCLR2.stdout,
        text=True,
        capture_output=True,
    )

    return Output_Lab_1.stdout, Output_Lab_2.stdout


def compare_slice(input_lab_1, input_lab_2, output_lab_1, output_lab_2):
    # takes the input lab's and the output and returns the delta comparisons

    input_lab_1 = format_data(input_lab_1)
    input_lab_2 = format_data(input_lab_2)
    output_lab_1 = format_data(output_lab_1)
    output_lab_2 = format_data(output_lab_2)

    # generate the deltas for the input vector
    # represent the data as vectors
    input_vectors = [[x, y] for x, y in zip(input_lab_1, input_lab_2)]
    # print(input_vectors)

    output_vectors = [[x, y] for x, y in zip(output_lab_1, output_lab_2)]
    # print(output_vectors)

    # find out deltaE, deltaL, deltaC and deltaH
    inputDeltas = get_all_deltas(input_vectors)
    outputDeltas = get_all_deltas(output_vectors)

    # print(inputDeltas)
    # print(outputDeltas)
    # get differences in deltas
    comparison = compare_deltas(inputDeltas, outputDeltas)
    return comparison


def compare_all(original_slices, converted_slices):
    compared_slices = []
    for i in range(len(original_slices)):
        original_slice1, original_slice2 = original_slices[i]
        converted_slice1, converted_slice2 = converted_slices[i]
        comparison = compare_slice(
            original_slice1, original_slice2, converted_slice1, converted_slice2
        )
        compared_slices.append(comparison)
    return compared_slices


# write_temp_files(lab_points, original_shape)
write_temp_files(filtered_points_nparray)


def process_files():
    converted_slices = []
    original_slices = []

    for i in range(62):

        slice1_path = f"temporary_files/slice1_{i}.txt"
        slice2_path = f"temporary_files/slice2_{i}.txt"

        with open(slice1_path, "r") as f1:
            original_slice_1 = f1.read()

        with open(slice2_path, "r") as f2:
            original_slice_2 = f2.read()

            original_slices.append((original_slice_1, original_slice_2))

        converted_slice_1, converted_slice_2 = take_files_from_folder_and_convert(
            slice1_path,
            slice2_path,
            profile,
        )

        converted_slices.append((converted_slice_1, converted_slice_2))
    return original_slices, converted_slices


original_slices, converted_slices = process_files()

compared_slices = compare_all(original_slices, converted_slices)
print(len(compared_slices))
print(compared_slices[0])


def save_comparisons_to_files(hashmaps, output_folder):
    """
    Saves each comparison in the list to a separate JSON file in the specified folder.

    Parameters:
        hashmaps (list): A list of hashmaps (dictionaries) to save.
        output_folder (str): The folder where the JSON files will be saved.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through the hashmaps and save each to a file
    for i, hashmap in enumerate(hashmaps):
        file_path = os.path.join(output_folder, f"hashmap_{i}.json")
        with open(file_path, "w") as json_file:
            json.dump(hashmap, json_file, indent=4)  # Save as pretty JSON
        print(f"Saved {file_path}")


save_comparisons_to_files(compared_slices, "comparisons dump")
