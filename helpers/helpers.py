import os


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


def clear_folder(folder_path):
    """Deletes all files in the specified folder but keeps the folder itself."""
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Delete files and symbolic links
            elif os.path.isdir(file_path):

                import shutil

                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
