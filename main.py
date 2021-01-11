from __future__ import annotations

import os
import tempfile
import subprocess
import warnings

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm


def read_dataset(filepath: str, interpolation_step: float = 1) -> pd.DataFrame:
    """
    Read dataset and interpolate equally spaced points.

    :param filepath: Relative path to file.
    :param interpolation_step: Optional. The interval at which to interpolate points.
    :returns: Interpolated dataset without original points.
    """
    dataset = pd.read_csv(filepath, delimiter="\t", header=None, names=["Easting", "Northing", "Height"])
    # Dataset is added a last point which equals the first data point to close the shape. Cummulative distance between first point and last point equals the polygon's circumference.
    dataset.loc[dataset.index.max() + 1] = dataset.iloc[0]

    # Calculate distances from first point and sets that as index.
    distances_from_first = np.cumsum(np.linalg.norm((dataset - dataset.shift(1)).fillna(0), axis=1))
    dataset.index = distances_from_first

    # Generate new indices pr. interpolation_step and compile the old and new indices in a list.
    new_indices = np.arange(0, dataset.index.max(), step=interpolation_step)
    all_indices = np.unique(np.r_[dataset.index, new_indices])

    # Re-index the dataset to include new and old indices. Interpolates the new coordinates from the old indices' coordinates.
    # Only includes new indices and coordinates.
    interpolated_dataset = dataset.reindex(all_indices).interpolate(method="linear").reindex(new_indices)
    return interpolated_dataset


def plot_from_xdir(dataset: pd.DataFrame):
    """
    Plot polygon from x-direction in northing/height graph.
    """
    plt.plot(dataset["Northing"], dataset["Height"])
    plt.show()


def plot_plane_and_layer(dataset: pd.DataFrame, model: LinearRegression):
    """
    Calculate estimated Z-values for plane and plot plane and polygon (AKA layer) in 3D view.
    """
    # ERIK: Imperative mood
    x_values = [dataset["Easting"].min(), dataset["Easting"].max()] * 2
    y_values = [dataset["Northing"].max()] * 2 + [dataset["Northing"].min()] * 2
    # Calculate predicted z values and put it in a variable.
    predicted_zvalues = model.predict(X=np.transpose([x_values, y_values]))

    ax = plt.axes(projection="3d")
    ax.plot3D(dataset["Easting"], dataset["Northing"], dataset["Height"])
    ax.plot_surface(np.array(x_values).reshape((2, 2)), np.array(
        y_values).reshape((2, 2)), predicted_zvalues.reshape((2, 2)), alpha=0.5)

    # The block below makes the axes equal, starting from the z/y/z-min, and ending at the maximum difference of
    # the dimensions. This means the axes are to scale with each other.
    maximum_offset = max(
        dataset["Easting"].max() - dataset["Easting"].min(),
        dataset["Northing"].max() - dataset["Northing"].min(),
        dataset["Height"].max() - dataset["Height"].min()
    )
    ax.set_xlim(dataset["Easting"].min(), dataset["Easting"].min() + maximum_offset)
    ax.set_ylim(dataset["Northing"].min(), dataset["Northing"].min() + maximum_offset)
    ax.set_zlim(dataset["Height"].min(), dataset["Height"].min() + maximum_offset)

    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.show()


def estimate_plane(dataset: pd.DataFrame):
    """
    Make flat plane based on polygon points.
    """
    model = LinearRegression()
    model.fit(X=dataset[["Easting", "Northing"]], y=dataset["Height"])
    # plot_plane_and_layer(dataset, model)
    return model


def calculate_plane_tilt(model: LinearRegression) -> tuple[float, float]:
    """
    Calculate plane tilt from estimated plane.
    :param model: estimated plane
    :returns: tilt-direction and tilt
    """
    # Try 1000 different angles around a circle.
    angles_to_try = np.linspace(0, np.pi * 2, num=1000)
    x_values_to_try = np.cos(angles_to_try)
    y_values_to_try = np.sin(angles_to_try)

    # Find elevations corresponding to the x,y-pairs in circle.
    elevations = model.predict(X=np.transpose([x_values_to_try, y_values_to_try]))

    # Find angles corresponding to lowest elevation, which is the same as tilt-direction. Converts from radians to degrees.
    lowest_index = np.argwhere(elevations == elevations.min())[0]
    tilt_direction = np.rad2deg(angles_to_try[lowest_index])

    # Checks if plane tilts very little. If yes, it is horizontal.
    if np.std(elevations) < 1e-4:
        return 0.0, 0.0

    # Tilt is calculated from inverse tangent of the elevation difference divided by diameter of the circle.
    tilt = np.rad2deg(np.arctan((elevations.max() - elevations.min()) / 2))

    return tilt_direction, tilt


def calculate_plane_x_y_angles(model: LinearRegression):
    """
    Calculate angles along the x,y axes from plane model.
    """
    y_angle = -np.arctan(np.diff(model.predict([[-1, 0], [1, 0]]))[0] / 2)
    x_angle = np.arctan(np.diff(model.predict([[0, -1], [0, 1]]))[0] / 2)

    if np.isnan(x_angle):

        raise AssertionError(f"x_angle is NaN!")

    return x_angle, y_angle


def pdal_rotate_dataset(dataset, x_angle, y_angle):
    # Copy dataset and make new pdal friendly column names.
    dataset_copy = dataset.rename(columns={"Easting": "X", "Northing": "Y", "Height": "Z"})
    # Make 0,0 point in middle of dataset.
    midpoint = dataset_copy.mean(axis=0)
    dataset_copy -= midpoint

    # Make temp-directory and define name for temp-dataset.
    temp_dir = tempfile.TemporaryDirectory()
    dataset_temp_filepath = os.path.join(temp_dir.name, "dataset.xyz")
    output_temp_filepath = os.path.join(temp_dir.name, "output.xyz")

    # Save temp-dataset in temp-directory above.
    dataset_copy.to_csv(dataset_temp_filepath, index=False)

    # Check if angles are NaN or actual values.
    if np.any(np.isnan([x_angle, y_angle])):
        raise AssertionError(f"NaN in x_angle or y_angle: {x_angle, y_angle}")

    pipeline = jinja2.Template("""
    [
        {
            "type": "readers.text",
            "filename": "{{ infile }}"
        },
        {
            "type": "filters.transformation",
            "matrix": "1 0 0 0 0 {{ cos_x }} {{ -sin_x }} 0 0 {{ sin_x }} {{ cos_x }} 0 0 0 0 1"
        },
        {
            "type": "filters.transformation",
            "matrix": "{{ cos_y }} 0 {{ sin_y }} 0 0 1 0 0 {{ -sin_y }} 0 {{ cos_y }} 0 0 0 0 1"
        },
        {
            "type": "writers.text",
            "filename": "{{ outfile }}"
        }
    ]
    """).render(dict(
        infile=dataset_temp_filepath.replace("\\", "/"),
        outfile=output_temp_filepath.replace("\\", "/"),
        cos_x=np.cos(x_angle),
        sin_x=np.sin(x_angle),
        cos_y=np.cos(y_angle),
        sin_y=np.sin(y_angle)
    ))
    subprocess.run(["pdal", "pipeline", "--stdin"], input=pipeline, encoding="utf-8", check=True)
    
    result = (pd.read_csv(output_temp_filepath) + midpoint).rename(columns={"X": "Easting", "Y": "Northing", "Z": "Height"})
   

    return result


def measure_width(dataset):

    distances = scipy.spatial.distance.cdist(
        dataset[["Easting", "Northing", "Height"]],
        dataset[["Easting", "Northing", "Height"]]
    )

    width = distances.max()

    return width


def get_polygon_filepaths(mountain_name):
    directory_name = f"Data/Polygons_{mountain_name}/"
    filepaths = []
    for file_name in os.listdir(directory_name):
        filepath = os.path.join(directory_name, file_name)
        filepaths.append(filepath)
    return filepaths

def get_lines_filepaths(mountain_name):
    directory_name = f"Data/Lines_{mountain_name}/"
    filepaths = []
    try:
        for file_name in os.listdir(directory_name):
            filepath = os.path.join(directory_name, file_name)
            filepaths.append(filepath)
    except FileNotFoundError:
        return filepaths
    return filepaths

def get_mountain_names():
    mountain_names = []
    for filename in os.listdir("Data"):
        mountain_name = filename.replace("Lines_","").replace("Polygons_","")
        mountain_names.append(mountain_name)
    unique_mountain_names = np.unique(mountain_names)
    return unique_mountain_names


def prepare_data():
    if os.path.isfile("all_layer_data.csv"):
        return pd.read_csv("all_layer_data.csv",index_col=0)
    data = pd.DataFrame(columns=["mountain_name","thickness","width","height","easting","northing"])
    mountain_names = get_mountain_names()
    count = 0
    for mountain_name in mountain_names:
        for filepath in tqdm(get_polygon_filepaths(mountain_name)):
            dataset = read_dataset(filepath)
            width = measure_width(dataset)
            rectified_dataset = dataset.copy()
            for _ in range(2):
                model = estimate_plane(rectified_dataset)
                x_angle, y_angle = calculate_plane_x_y_angles(model)
                rectified_dataset = pdal_rotate_dataset(rectified_dataset, -x_angle, -y_angle)

            model = estimate_plane(rectified_dataset)

            _, tilt = calculate_plane_tilt(model)
            # print(f"Tilt: {tilt:.2f} degrees")

            # plot_plane_and_layer(rectified_dataset, model)

            thickness = rectified_dataset["Height"].max() - rectified_dataset["Height"].min()

            #if thickness > 20:
            #    plot_plane_and_layer(rectified_dataset, model)

            # print(f"Layer {i} is {thickness:.2f} m thick\n")

            height = np.median(dataset["Height"])
            easting = np.median(dataset["Easting"])
            northing = np.median(dataset["Northing"])
            data.loc[count] = mountain_name, thickness, width, height, easting, northing
            count += 1
    data.to_csv("all_layer_data.csv")
    return data


def plot_data():
    row = 0
    mountain_names = get_mountain_names()
    data = prepare_data()

    for mountain_name, mountain_data in data.groupby("mountain_name"):
        row += 1
        line_heights = []
        line_widths = []
        for filepath in get_lines_filepaths(mountain_name):
            dataset = read_dataset(filepath)
            width = measure_width(dataset)
            line_widths.append(width)
            line_heights.append(np.median(dataset["Height"]))

        plt.subplot(len(mountain_names), 3, 1 + (3 * (row - 1)))
        plt.scatter(mountain_data["width"], mountain_data["height"], facecolor="darkslategrey", alpha=0.7)
        plt.scatter(line_widths, line_heights, facecolor="None", edgecolor=(0.38,0.26,0.98,.5))

        plt.xlabel("Layer width (m)", fontsize=12)
        if row == 3:
            plt.ylabel("Elevation (m a.s.l.)", fontsize=12)

        plt.subplot(len(mountain_names), 3, 2 + (3 * (row - 1)))
        nice_label = mountain_name.replace("oe","ø").replace("_"," ")
        plt.text(0.5, 0.8, nice_label, transform=plt.gca().transAxes, ha="center", fontsize=12)
        plt.scatter(mountain_data["thickness"], mountain_data["height"], facecolor="darkslategrey", alpha=0.7)

        plt.xlabel("Layer thickness (m)", fontsize=12)
        if row == 3:
            plt.ylabel("Elevation (m a.s.l.)", fontsize=12)

        plt.subplot(len(mountain_names), 3, 3 + (3 * (row - 1)))
        plt.scatter(mountain_data["width"], mountain_data["thickness"], facecolor="darkslategrey", alpha=0.7)

        plt.xlabel("Layer width (m)", fontsize=12)
        if row == 3:
            plt.ylabel("Layer thickness (m)", fontsize=12)
        
        

    plt.show()

def plot_width_thickness():
    data = prepare_data()

    model = RANSACRegressor()
    model.fit(data["width"].values.reshape(-1, 1), data["thickness"])
    linear_width_values = np.linspace(data["width"].min(), data["width"].max())
    predicted_thicknesses = model.predict(linear_width_values.reshape(-1, 1))
    confidence_interval = 1.96 * np.std(predicted_thicknesses)/np.mean(predicted_thicknesses)

    for i, (mountain_name, mountain_data) in enumerate(data.groupby("mountain_name")):
        nice_label = mountain_name.replace("oe","ø").replace("_"," ")
        color = plt.get_cmap("twilight")(i / len(np.unique(data["mountain_name"])))

        plt.scatter(mountain_data["width"],mountain_data["thickness"],label=nice_label, c=(color,), edgecolors="darkgrey")
       
    plt.plot(linear_width_values, predicted_thicknesses)
    plt.fill_between(linear_width_values, predicted_thicknesses + confidence_interval, predicted_thicknesses - confidence_interval, color="moccasin", alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #plot_width_thickness()
    #print(prepare_data())
    plot_data()