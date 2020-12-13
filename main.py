from __future__ import annotations

import os
import tempfile
import warnings

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdal
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression, RANSACRegressor
from tqdm import tqdm


def read_dataset(filepath: str, interpolation_step: float = 1) -> pd.DataFrame:
    """
    Reads dataset and interpolates equally spaced points.

    :param filepath: Relative path to file.
    :param interpolation_step: Optional. The interval at which to interpolate points.
    :returns: Interpolated dataset without original points.
    """
    # ERIK: Docstrings should start in imperative mood: "Read" instead of "Reads"
    dataset = pd.read_csv(filepath, delimiter="\t", header=None, names=["Easting", "Northing", "Height"])
    # Dataset is added a last point which equals the first data point to close the shape. This equals the polygon's circumference.
    # ERIK: What equals the polygon's circumference?
    dataset.loc[dataset.index.max() + 1] = dataset.iloc[0]

    # Calculates distances from first point and sets that as index.
    distances_from_first = np.cumsum(np.linalg.norm((dataset - dataset.shift(1)).fillna(0), axis=1))
    dataset.index = distances_from_first

    # Generates new indices pr. interpolation_step and compiles the old and new indices in a list.
    new_indices = np.arange(0, dataset.index.max(), step=interpolation_step)
    all_indices = np.unique(np.r_[dataset.index, new_indices])

    # Re-index the dataset to include new and old indices. Interpolates the new coordinates from the old indices' coordinates.
    #  Only includes new indices and coordinates.
    interpolated_dataset = dataset.reindex(all_indices).interpolate(method="linear").reindex(new_indices)
    return interpolated_dataset


def plot_from_xdir(dataset: pd.DataFrame):
    """
    Plots polygon from x-direction in northing/height graph.
    """
    # ERIK: Imperative mood
    plt.plot(dataset["Northing"], dataset["Height"])
    plt.show()


def plot_plane_and_layer(dataset: pd.DataFrame, model: LinearRegression):
    """
    Calculates estimated Z-values for plane and plots plane and polygon (AKA layer) in 3D view.
    """
    # ERIK: Imperative mood
    x_values = [dataset["Easting"].min(), dataset["Easting"].max()] * 2
    y_values = [dataset["Northing"].max()] * 2 + [dataset["Northing"].min()] * 2
    # Calculates predicted z values and makes variable.
    # ERIK: Makes what variable? (we instantiate a 3D axis (an axis in matplotlib means a canvas))
    predicted_zvalues = model.predict(X=np.transpose([x_values, y_values]))
    ax = plt.axes(projection="3d")

    ax.plot3D(dataset["Easting"], dataset["Northing"], dataset["Height"])
    ax.plot_surface(np.array(x_values).reshape((2, 2)), np.array(
        y_values).reshape((2, 2)), predicted_zvalues.reshape((2, 2)), alpha=0.5)

    # ERIK: The block below makes the axes equal, starting from the z/y/z-min, and ending at the maximum difference of
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
    Makes flat plane based on polygon points.
    """
    # ERIK: Imperative mood
    model = LinearRegression()
    model.fit(X=dataset[["Easting", "Northing"]], y=dataset["Height"])
    # plot_plane_and_layer(dataset, model)
    return model


def calculate_plane_tilt(model: LinearRegression) -> tuple[float, float]:

    angles_to_try = np.linspace(0, np.pi * 2, num=1000)
    x_values_to_try = np.cos(angles_to_try)
    y_values_to_try = np.sin(angles_to_try)

    elevations = model.predict(X=np.transpose([x_values_to_try, y_values_to_try]))

    lowest_index = np.argwhere(elevations == elevations.min())[0]

    tilt_direction = np.rad2deg(angles_to_try[lowest_index])

    if np.std(elevations) < 1e-4:
        return 0.0, 0.0

    tilt = np.rad2deg(np.arctan((elevations.max() - elevations.min()) / 2))

    return tilt_direction, tilt


def calculate_plane_x_y_angles(model: LinearRegression):

    x_angle = -np.arctan(np.diff(model.predict([[-1, 0], [1, 0]]))[0] / 2)
    y_angle = np.arctan(np.diff(model.predict([[0, -1], [0, 1]]))[0] / 2)

    if np.isnan(x_angle):

        raise AssertionError(f"x_angle is NaN!")

    return x_angle, y_angle


def pdal_rotate_dataset(dataset, x_angle, y_angle):

    dataset_copy = dataset.rename(columns={"Easting": "X", "Northing": "Y", "Height": "Z"})

    midpoint = dataset_copy.mean(axis=0)

    dataset_copy -= midpoint

    temp_dir = tempfile.TemporaryDirectory()
    dataset_temp_filepath = os.path.join(temp_dir.name, "dataset.xyz")
    grid_cloud_temp_filepath = os.path.join(temp_dir.name, "grid.xyz")

    dataset_copy.to_csv(dataset_temp_filepath, index=False)

    if np.any(np.isnan([x_angle, y_angle])):
        raise AssertionError(f"NaN in x_angle or y_angle: {x_angle, y_angle}")

    pipeline = pdal.Pipeline(jinja2.Template("""
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
        }
    ]
    """).render(dict(
        infile=dataset_temp_filepath,
        cos_x=np.cos(y_angle),
        sin_x=np.sin(y_angle),
        cos_y=np.cos(x_angle),
        sin_y=np.sin(x_angle)
    )))

    pipeline.execute()

    result = (pd.DataFrame(pipeline.arrays[0]) +
              midpoint).rename(columns={"X": "Easting", "Y": "Northing", "Z": "Height"})

    return result


def rectify_dataset(input_dataset: pd.DataFrame, n_iterations=50) -> pd.DataFrame:
    raise DeprecationWarning("Function should not need to be used")
    dataset = input_dataset.copy()

    def error_minimisation(dataset, learning_rate_factor=4):
        model = estimate_plane(dataset)
        _, tilt = calculate_plane_tilt(model)

        sign = 1
        for i in range(n_iterations):
            x_angle, y_angle = calculate_plane_x_y_angles(model)
            if tilt < 0.15:
                break

            learning_rate = min(tilt * learning_rate_factor, 1)
            assert ~np.isnan(tilt), "Tilt is NaN!"
            new_dataset = pdal_rotate_dataset(
                dataset=dataset,
                x_angle=sign * x_angle * learning_rate,
                y_angle=sign * y_angle * learning_rate
            )

            new_model = estimate_plane(new_dataset)
            _, new_tilt = calculate_plane_tilt(new_model)

            if new_tilt < tilt:
                model = new_model
                dataset = new_dataset
                tilt = new_tilt
            else:
                sign *= -1
        else:
            raise ValueError(f"Rectification never converged after {n_iterations} iterations. Tilt: {tilt:.2f} degrees")

        return dataset

    factors_to_try = np.linspace(0.6, 2, num=15) ** 10
    np.random.shuffle(factors_to_try)
    for factor in factors_to_try:
        try:
            rectified_dataset = error_minimisation(dataset, learning_rate_factor=factor)
            break
        except ValueError:
            continue
    else:
        raise ValueError("Rectification failed")

    return rectified_dataset


if __name__ == "__main__":

    thicknesses = []
    heights = []
    for i in range(1, 68):
        dataset = read_dataset(f"Data/Polygons_Litledalsfjellet/SS_{i}.asc")

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

        if thickness > 20:
            plot_plane_and_layer(rectified_dataset, model)

        # print(f"Layer {i} is {thickness:.2f} m thick\n")

        thicknesses.append(thickness)
        heights.append(np.median(dataset["Height"]))

    n_bins = 10
    height_interval = np.linspace(np.min(heights), np.max(heights), num=n_bins)
    bins = np.digitize(heights, height_interval)

    maxvals = []
    for i in range(n_bins):
        vals = np.array(thicknesses)[bins == i]
        maxval = vals.max() if vals.shape[0] != 0 else np.nan
        maxvals.append(maxval)

    model = RANSACRegressor(residual_threshold=4)
    model.fit(height_interval[~np.isnan(maxvals)].reshape(-1, 1), np.array(maxvals)[~np.isnan(maxvals)])

    plt.plot(model.predict(height_interval.reshape(-1, 1)), height_interval)
    plt.scatter(thicknesses, heights)
    # plt.scatter(np.array(maxvals)[~np.isnan(maxvals)][model.inlier_mask_],
    #            height_interval[~np.isnan(maxvals)][model.inlier_mask_], s=4)

    plt.xlabel("Layer thickness (m)")
    plt.ylabel("Elevation (m a.s.l.)")

    plt.show()
