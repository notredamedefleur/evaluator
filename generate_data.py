# here we generate the vector lists for further evaluation in evaluate.py
import os
import subprocess

from colour import Lab_to_XYZ, XYZ_to_RGB
import numpy as np
from colour.models import RGB_COLOURSPACE_ADOBE_RGB1998


# this is a temp quickfix, because it doesnt want to import for some reason
folder_path_input = "Evaluate_RGB_Profiles/Input/"
folder_path_output = "Evaluate_RGB_Profiles/Output/"

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


# set boundaries (numbers are floats)
L_MIN, L_MAX = 0.0, 100.0
A_MIN, A_MAX = -128.0, 127.0
B_MIN, B_MAX = -128.0, 127.0

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


# grid = generate_lab_grid(L_MIN, L_MAX, A_MIN, A_MAX, B_MIN, B_MAX, step)


def filter_to_adobe_rgb(vectors):
    # takes a list of vectors, returns ones that fit into the adobe rgb space
    # algo: convert lab to xyz using the d65, then convert to rgb using the matrix and check if lies inside
    filtered_vectors = []
    for vector in vectors:
        # if the starting point is within, then good
        if is_within_adobe_rgb(*vector[0]):
            filtered_vectors.append(vector)
    return filtered_vectors


# same func but for the whole generated points
def is_within_adobe_rgb(lab_points):
    # func for an array of points
    xyz_points = Lab_to_XYZ(lab_points)
    rgb_points = XYZ_to_RGB(
        xyz_points,
        illuminant_XYZ="D65",
        illuminant_RGB=RGB_COLOURSPACE_ADOBE_RGB1998.whitepoint,
        RGB_COLOURSPACE=RGB_COLOURSPACE_ADOBE_RGB1998,
        colourspace="Adobe RGB (1998)",
    )
    return np.all((rgb_points >= 0) & (rgb_points <= 1), axis=1)


# this func uses our script to check if lab point is in gamut
def is_lab_point_in_gamut(point: str):
    # step 1: take the profile and run the script
    if os.path.isdir(folder_path_input):
        for file_name in os.listdir(folder_path_input):
            if file_name.endswith(".icc"):
                inputfile = folder_path_input + file_name

                dstP = folder_path_input + os.path.splitext(file_name)[0] + ".dstP"
                inpP = folder_path_input + os.path.splitext(file_name)[0] + ".inpP"

                subprocess.run(["scripts/icc2tags", inputfile, "dstP"])
                subprocess.run(["scripts/icc2tags", inputfile, "inpP"])

                pointInRGB = subprocess.run(
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
                    input=point,
                    text=True,
                    capture_output=True,
                )

                outputCLR = subprocess.run(  # i forgot what clr means but ok
                    ["scripts/transiccgmg.exe", "-l", inputfile, "-c", "0", "-n"],
                    input=str(pointInRGB),
                    text=True,
                    capture_output=True,
                )

                # so this is the end point of the rundabout
                outputLAB = subprocess.run(
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
                    input=str(outputCLR),
                    text=True,
                    capture_output=True,
                )

                # step 2 run the script backwards

                pointInRGBBack = subprocess.run(
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
                    input=str(outputLAB),
                    text=True,
                    capture_output=True,
                )

                outputCLRBack = subprocess.run(  # i forgot what clr means but ok
                    ["scripts/transiccgmg.exe", "-l", inputfile, "-c", "0", "-n"],
                    input=str(pointInRGBBack),
                    text=True,
                    capture_output=True,
                )

                # so this is the end point of the rundabout
                outputLABBack = subprocess.run(
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
                    input=str(outputCLRBack),
                    text=True,
                    capture_output=True,
                )

                # compare
                # print("enter value: " + point)
                # print("exit value: " + outputLABBack.stdout)
                return


is_lab_point_in_gamut("50, 0, 0")  # doesnt seem to work. the disparity is too big.


def filter_adobe_rgb_points(lab_points):

    mask = is_within_adobe_rgb(lab_points)
    # returns the points in adobe rgb
    return lab_points[mask]


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


testVectors = [
    [[50.0, 0.0, 0.0], [10.0, -128.0, -128.0]],
    [[100.0, 90.0, 90.0], [20.0, -128.0, -128.0]],
]


# the function generate_lab_grid returns two things: the grid in a numpy 3d array (lab_grid)
# and the grid in a flattened 1d array (lab_points)
# this will be fixed later, but now some functions use the 3d grid and some the flattened one
lab_grid, lab_points = generate_lab_grid(L_MIN, L_MAX, A_MIN, A_MAX, B_MIN, B_MAX, step)
# print(lab_grid)


# Filter points within Adobe RGB
filtered_points = filter_adobe_rgb_points(lab_points)

# print(f"Total points: {lab_points.shape[0]}")
# print(f"Points within Adobe RGB: {filtered_points.shape[0]}")
# print("Sample filtered points:")
# print(filtered_points[:10])


# sample visualisation for testing. looks ok to me, but not sure if everything is correct.
# but the plot does kind of like the 1998 gamut
# visualize_lab_points_plotly(filtered_points)
