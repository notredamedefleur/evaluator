# here we generate the vector lists for further evaluation in evaluate.py

import os
import subprocess
import plotly.graph_objects as go

# this is a temp quickfix, because it doesnt want to import for some reason
folder_path_input = "Evaluate_RGB_Profiles/Input/"
folder_path_output = "Evaluate_RGB_Profiles/Output/"

# change profile here
profile = "CS_sep_AdobeRGB_2_ISOcoatedv2-39L_TAC330_V6.icc"


# set boundaries (numbers are floats)
L_MIN, L_MAX = 0.0, 100.0
A_MIN, A_MAX = -120.0, 120.0
B_MIN, B_MAX = -120.0, 120.0

step = 10  # 10 for now, can change later

import numpy as np


def generate_lab_grid(L_min, L_max, A_min, A_max, B_min, B_max, step):
    """
    Generates a 3D grid of LAB values and returns both a 3D array and a flattened 1D array.

    Parameters:
    - L_min, L_max (float): Range for L values.
    - A_min, A_max (float): Range for A values.
    - B_min, B_max (float): Range for B values.
    - step (float): Step size for the grid.

    Returns:
    - lab_grid (numpy.ndarray): 3D array of LAB values.
    - lab_points (numpy.ndarray): Flattened 1D array of LAB values.
    """
    # Generate ranges for L, A, and B
    L_values = np.arange(L_min, L_max + step, step)
    L_values[-1] = L_max  # Ensure the last value is exactly L_max
    A_values = np.arange(A_min, A_max + step, step)
    A_values[-1] = A_max  # Ensure the last value is exactly A_max
    B_values = np.arange(B_min, B_max + step, step)
    B_values[-1] = B_max  # Ensure the last value is exactly B_max

    # Create the 3D grid
    L_grid, A_grid, B_grid = np.meshgrid(L_values, A_values, B_values, indexing="ij")
    lab_grid = np.stack([L_grid, A_grid, B_grid], axis=-1)  # 3D array of LAB values

    # Flatten the grid into a 1D array of LAB points
    lab_points = lab_grid.reshape(-1, 3)  # Flattened array with LAB points

    return lab_grid, lab_points


def arr_to_string(arr: np.ndarray):
    return "\n".join(" ".join(map(str, point)) for point in arr)


def clip_values(data_str):
    """
    Clips all values in the input data to a maximum of 255.

    Parameters:
    data_str (str): A space-separated string with new lines representing rows.

    Returns:
    np.ndarray: The clipped array where values greater than 255 are set to 255.
    """
    data_list = [
        list(map(float, line.split())) for line in data_str.strip().split("\n")
    ]
    data_array = np.array(data_list)  # Convert to numpy array
    return np.clip(data_array, None, 255)


def compare_points(points1, points2, tolerance=0.1):
    """
    Compares two lists of space-separated points and filters out points
    that are not within the tolerance for all values.
    """
    filtered_points = []

    for i, (point1, point2) in enumerate(zip(points1, points2), start=1):
        values1 = list(map(float, point1.strip().split()))
        values2 = list(map(float, point2.strip().split()))
        # print(values1, values2)

        # Check if all values in the point are within the tolerance
        if all(abs(v1 - v2) < tolerance for v1, v2 in zip(values1, values2)):
            filtered_points.append(point1)  # Keep the original point

    return filtered_points


def filter_lab_points(lab_points, profile):
    # step 1 -- take all the points and put them into a space separated list
    lab_points = arr_to_string(lab_points)

    # step 2 -- run the list thru the script

    profile_path = folder_path_input + profile
    dstP = folder_path_input + os.path.splitext(profile)[0] + ".dstP"
    inpP = folder_path_input + os.path.splitext(profile)[0] + ".inpP"

    subprocess.run(["scripts/icc2tags", profile_path, "dstP"])
    subprocess.run(["scripts/icc2tags", profile_path, "inpP"])

    points_in_rgb = subprocess.run(
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
        input=lab_points,
        text=True,
        capture_output=True,
    )

    # take all rgb's higher than 255 and clip em to 255
    points_in_rgb = clip_values(points_in_rgb.stdout)
    points_in_rgb = arr_to_string(points_in_rgb)

    points_to_compare = subprocess.run(
        [
            "scripts/transiccgmg.exe",
            "-o",
            "*Lab",
            "-i",
            inpP,
            "-t",
            "3",
            "-c",
            "0",
            "-n",
        ],
        input=str(points_in_rgb),
        text=True,
        capture_output=True,
    )
    lab_points_list = lab_points.strip().split("\n")

    points_to_compare_list = points_to_compare.stdout.strip().split("\n")

    return compare_points(lab_points_list, points_to_compare_list)


def points_to_nparray(points: list[str]) -> np.ndarray:
    """
    Converts a list of space-separated LAB points into a 2D NumPy array.

    Parameters:
    - points (list[str]): A list where each entry represents a LAB point (L, a, b).

    Returns:
    - np.ndarray: A 2D NumPy array of shape (n, 3), where each row represents a LAB point.
    """
    return np.array([list(map(float, point.split())) for point in points])


def points_to_3d_nparray(points: list[str]) -> np.ndarray:
    """
    Converts a list of space-separated LAB points into a 3D NumPy array.

    Parameters:
    - points (list[str]): A list where each entry represents a LAB point (x, y, z coordinates).

    Returns:
    - np.ndarray: A NumPy array representing the points in 3D space.
    """
    data_array = np.array([list(map(float, point.split())) for point in points])

    # Determine grid dimensions by finding unique coordinates
    x_vals = np.unique(data_array[:, 0])
    y_vals = np.unique(data_array[:, 1])
    z_vals = np.unique(data_array[:, 2])

    x_dim, y_dim, z_dim = len(x_vals), len(y_vals), len(z_vals)
    shape = (x_dim, y_dim, z_dim, 3)

    # Create an empty grid
    grid = np.zeros(shape)

    # Populate the grid with points
    for point in data_array:
        x_idx = np.where(x_vals == point[0])[0][0]
        y_idx = np.where(y_vals == point[1])[0][0]
        z_idx = np.where(z_vals == point[2])[0][0]
        grid[x_idx, y_idx, z_idx] = point

    return grid


# temporary visualisation function
def visualize_lab_points_plotly(lab_points):
    """
    Visualize LAB points in a 3D scatter plot using Plotly.

    Parameters:
        lab_points (numpy.ndarray): Array of LAB points with shape (N, 3).
    """
    # Extract L, A, and B components
    L = lab_points[:, 0]
    A = lab_points[:, 1]
    B = lab_points[:, 2]

    # Create a 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=A,  # A axis (Green-Red)
                y=B,  # B axis (Blue-Yellow)
                z=L,  # L axis (Lightness)
                mode="markers",
                marker=dict(
                    size=3,
                    color=L,  # Color points by Lightness
                    colorscale="Viridis",  # Colormap
                    opacity=0.7,
                ),
            )
        ]
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="A (Green-Red)",
            yaxis_title="B (Blue-Yellow)",
            zaxis_title="L (Lightness)",
        ),
        title="LAB Color Space Points",
        margin=dict(l=0, r=0, b=0, t=50),
    )

    # Show the plot
    fig.show()


lab_grid, lab_points = generate_lab_grid(L_MIN, L_MAX, A_MIN, A_MAX, B_MIN, B_MAX, step)


filtered_points = filter_lab_points(lab_points, profile)

filtered_points_nparray = points_to_nparray(filtered_points)

# uncomment this to show 3d plot
# visualize_lab_points_plotly(points_to_nparray(filtered_points))
