import math
import os
import shutil
import subprocess
import numpy as np
import json
from helpers import format_data, clear_folder
from generate_data import profile

# here we take the data and evaluate it, dumping the diff. in deltas to comparisons dump

# import generated and filtered data to use here
from generate_data import (
    filtered_points_nparray,
)

folder_path_input = "Evaluate_RGB_Profiles/Input/"
folder_path_output = "Evaluate_RGB_Profiles/Output/"


profile_path = os.path.join(folder_path_input, profile)


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

        deltaH_squared = (deltaE**2) - (deltaL**2) - (deltaC**2)
        deltaH = math.sqrt(max(0.0, deltaH_squared))  # Ensure non-negative value
        delta_H_arr.append(deltaH)

    return {
        "deltaE": delta_E_arr,
        "deltaL": delta_L_arr,
        "deltaC": delta_C_arr,
        "deltaH": delta_H_arr,
    }


def compare_deltas(deltas1, deltas2):
    comparison = {}
    # this filters out the mismatching vectors
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


def extract_neighboring_slices(lab_points: np.ndarray):
    """
    Extract slices along the L, a, and b axes from a 2D NumPy array of LAB points.

    Parameters:
        lab_points (np.ndarray): A 2D NumPy array of shape (n, 3) representing LAB points.

    Returns:
        dict: A dictionary with keys 'L', 'a', and 'b' containing lists of slice pairs for each axis.
    """
    slices = {"L": [], "a": [], "b": []}

    # Extract slices along L-axis
    unique_L_values = np.unique(lab_points[:, 0])
    for i in range(len(unique_L_values) - 1):
        slice1 = lab_points[lab_points[:, 0] == unique_L_values[i]]
        slice2 = lab_points[lab_points[:, 0] == unique_L_values[i + 1]]
        slices["L"].append((slice1, slice2))

    # Extract slices along a-axis
    unique_a_values = np.unique(lab_points[:, 1])
    for i in range(len(unique_a_values) - 1):
        slice1 = lab_points[lab_points[:, 1] == unique_a_values[i]]
        slice2 = lab_points[lab_points[:, 1] == unique_a_values[i + 1]]
        slices["a"].append((slice1, slice2))

    # Extract slices along b-axis
    unique_b_values = np.unique(lab_points[:, 2])
    for i in range(len(unique_b_values) - 1):
        slice1 = lab_points[lab_points[:, 2] == unique_b_values[i]]
        slice2 = lab_points[lab_points[:, 2] == unique_b_values[i + 1]]
        slices["b"].append((slice1, slice2))

    return slices


def write_temp_files(lab_points, temp_folder="temporary_files"):
    """
    Writes neighboring slices as temporary files into corresponding subfolders.

    Parameters:
        lab_points (np.ndarray): The flattened LAB grid.
        temp_folder (str): The folder to save the temporary files.

    Returns:
        list of str: Paths to the created temporary files.
    """
    # Ensure the folder exists
    os.makedirs(temp_folder, exist_ok=True)

    # Extract neighboring slices
    neighboring_slices = extract_neighboring_slices(lab_points)
    print(
        "Number of neighboring slices:",
        sum(len(v) for v in neighboring_slices.values()),
    )

    # Store the paths of the created files
    file_paths = []

    for axis, slice_pairs in neighboring_slices.items():
        axis_folder = os.path.join(temp_folder, f"{axis}_axis")
        os.makedirs(axis_folder, exist_ok=True)

        # clear stuff before generating
        clear_folder(axis_folder)

        for i, (slice1, slice2) in enumerate(slice_pairs):
            file_path1 = os.path.join(axis_folder, f"slice1_{i}.txt")
            file_path2 = os.path.join(axis_folder, f"slice2_{i}.txt")

            np.savetxt(file_path1, slice1, delimiter="\t", fmt="%.2f")
            np.savetxt(file_path2, slice2, delimiter="\t", fmt="%.2f")

            file_paths.extend([file_path1, file_path2])

    print(f"Temporary files written to {temp_folder}")
    return file_paths


def convert(Input_Lab1, Input_Lab2, profile_path):

    print("start")

    inputfile = profile_path
    print(inputfile)
    outputfile = folder_path_output + os.path.splitext(profile_path)[0] + ".txt"
    dstP = os.path.splitext(profile_path)[0] + ".dstP"
    print(dstP)
    inpP = os.path.splitext(profile_path)[0] + ".inpP"
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


def compare(original_slices, converted_slices):
    compared_slices = []

    # Ensure the slices are structured correctly
    if len(original_slices) % 2 != 0 or len(converted_slices) % 2 != 0:
        raise ValueError("Mismatched slice pairs in original or converted slices.")

    for i in range(0, len(original_slices), 2):
        original_slice1 = original_slices[i]
        original_slice2 = original_slices[i + 1]
        converted_slice1 = converted_slices[i]
        converted_slice2 = converted_slices[i + 1]

        comparison = compare_slice(
            original_slice1, original_slice2, converted_slice1, converted_slice2
        )

        comparison["original_vector_entrance"] = format_data(original_slice1)
        comparison["converted_vector_entrance"] = format_data(converted_slice1)

        compared_slices.append(comparison)

    return compared_slices


def process_files(profile_path, folder):
    temp_folder = folder
    converted_slices = []
    original_slices = []

    # Get slice1_X and slice2_X files separately
    slice1_files = sorted(
        f
        for f in os.listdir(temp_folder)
        if f.startswith("slice1_") and f.endswith(".txt")
    )
    slice2_files = sorted(
        f
        for f in os.listdir(temp_folder)
        if f.startswith("slice2_") and f.endswith(".txt")
    )

    # Ensure the number of files match
    if len(slice1_files) != len(slice2_files):
        raise ValueError(
            f"Mismatched number of 'slice1' and 'slice2' files in {temp_folder}."
        )

    # Process files in pairs: "slice1_X" with "slice2_X"
    for file1, file2 in zip(slice1_files, slice2_files):
        file_path1 = os.path.join(temp_folder, file1)
        file_path2 = os.path.join(temp_folder, file2)

        with open(file_path1, "r") as f1, open(file_path2, "r") as f2:
            original_slice1 = f1.read()
            original_slice2 = f2.read()

        # Convert slices
        converted_slice1, converted_slice2 = convert(
            file_path1, file_path2, profile_path
        )

        # Extend slices
        original_slices.extend([original_slice1, original_slice2])
        converted_slices.extend([converted_slice1, converted_slice2])

    return original_slices, converted_slices


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
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty directories
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        os.makedirs(output_folder, exist_ok=True)

    # Loop through the hashmaps and save each to a file
    for i, hashmap in enumerate(hashmaps):
        file_path = os.path.join(output_folder, f"hashmap_{i}.json")
        with open(file_path, "w") as json_file:
            json.dump(hashmap, json_file, indent=4)  # Save as pretty JSON


def collect_all_comparisons(*args):
    all_comparisons = []

    for comparison_list in args:
        all_comparisons.extend(comparison_list)

    output_folder = "comparisons dump/collected_files"
    os.makedirs(output_folder, exist_ok=True)

    file_path = os.path.join(output_folder, "all_comparisons.json")
    with open(file_path, "w") as f:
        json.dump(all_comparisons, f, indent=4)


write_temp_files(filtered_points_nparray)


original_slices_a, converted_slices_a = process_files(
    profile_path, "temporary_files/a_axis"
)

original_slices_b, converted_slices_b = process_files(
    profile_path, "temporary_files/b_axis"
)
original_slices_l, converted_slices_l = process_files(
    profile_path, "temporary_files/l_axis"
)


compared_slices_a = compare(original_slices_a, converted_slices_a)
compared_slices_b = compare(original_slices_b, converted_slices_b)
compared_slices_l = compare(original_slices_l, converted_slices_l)


save_comparisons_to_files(compared_slices_a, "comparisons dump/a_axis")
save_comparisons_to_files(compared_slices_b, "comparisons dump/b_axis")
save_comparisons_to_files(compared_slices_l, "comparisons dump/l_axis")


def reshuffle_files_by_axis_from_json(folder_path, sort_axis="L"):
    """
    Reorders JSON files in a folder based on increasing values of `original_vector_entrance`.
    The order is determined by the specified axis (L, A, or B).

    Parameters:
        folder_path (str): Path to the folder containing JSON files.
        sort_axis (str): Axis to sort by ("L", "A", or "B").

    Returns:
        None
    """
    axis_index = {"L": 0, "A": 1, "B": 2}[sort_axis]
    file_data = []

    # Read each JSON file and extract the sorting key
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                # Extract the first vector from `original_vector_entrance`
                if (
                    "original_vector_entrance" in data
                    and data["original_vector_entrance"]
                ):
                    sort_value = data["original_vector_entrance"][0][axis_index]
                    file_data.append((sort_value, filename))
                else:
                    print(
                        f"Skipping {filename}: Missing 'original_vector_entrance' or invalid format."
                    )

    # Sort files by the extracted axis value
    file_data.sort(key=lambda x: x[0])

    # Rename files in the order of the sorted data
    for i, (_, filename) in enumerate(file_data):
        old_path = os.path.join(folder_path, filename)
        new_name = f"sorted_{i}.json"
        new_path = os.path.join(folder_path, new_name)

        # Rename the file to reflect the sorted order
        shutil.move(old_path, new_path)

    print(f"Files in {folder_path} have been reshuffled based on the {sort_axis}-axis.")


# this is a kind of workaround, but should not break anything. this can be discussed in refactoring
reshuffle_files_by_axis_from_json("comparisons dump/a_axis", sort_axis="A")
reshuffle_files_by_axis_from_json("comparisons dump/b_axis", sort_axis="B")
reshuffle_files_by_axis_from_json("comparisons dump/l_axis", sort_axis="L")


collect_all_comparisons(compared_slices_l, compared_slices_a, compared_slices_b)
