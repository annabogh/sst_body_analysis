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
import boundary_height
import rasterio as rio
import seaborn as sns


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
    all_layer_data = "cache/all_layer_data.csv"
    if os.path.isfile(all_layer_data):
        return pd.read_csv(all_layer_data,index_col=0)

    boundary_height_data = rio.open("boundary_height_buffered.tif")
    data = pd.DataFrame(columns=["mountain_name","thickness","width","height_asl", "height_abatt","easting","northing"])
    mountain_names = get_mountain_names()
    count = 0
    planes = estimate_datum_height()
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

            easting = np.median(dataset["Easting"])
            northing = np.median(dataset["Northing"])
            nice_label = mountain_name.replace("oe","ø").replace("_S","")
            batt_height = planes[nice_label].predict(np.reshape([easting, northing], (1, -1)))[0]
            
            height_asl = np.median(dataset["Height"])
            height_abatt = height_asl - batt_height


            data.loc[count] = mountain_name, thickness, width, height_asl, height_abatt, easting, northing
            count += 1
    data.to_csv(all_layer_data)
    return data


def plot_data():
    row = 0
    mountain_names = get_mountain_names()
    data = prepare_data()
    planes = estimate_datum_height()
    for mountain_name, mountain_data in data.groupby("mountain_name"):
        row += 1
        line_heights = []
        line_widths = []
        #boundary_height_data = rio.open("boundary_height_buffered.tif")
        for filepath in get_lines_filepaths(mountain_name):
            dataset = read_dataset(filepath)
            width = measure_width(dataset)
            line_widths.append(width)
            easting = np.median(dataset["Easting"])
            northing = np.median(dataset["Northing"])
            nice_label = mountain_name.replace("oe","ø").replace("_S","")
            batt_height = planes[nice_label].predict(np.reshape([easting, northing], (1, -1)))[0]
            
            height_asl = np.median(dataset["Height"])
            height_abatt = height_asl - batt_height
            line_heights.append(height_abatt)

        plt.subplot(len(mountain_names), 3, 1 + (3 * (row - 1)))
        plt.scatter(mountain_data["width"], mountain_data["height_abatt"], facecolor="darkslategrey", alpha=0.7)
        plt.scatter(line_widths, line_heights, facecolor="None", edgecolor=(0.38,0.26,0.98,.5))

        plt.xlabel("Layer width (m)", fontsize=12)
        if row == 3:
            plt.ylabel("Elevation (m above datum)", fontsize=12)

        plt.subplot(len(mountain_names), 3, 2 + (3 * (row - 1)))
        nice_label = mountain_name.replace("oe","ø").replace("_"," ")
        plt.text(0.5, 0.8, nice_label, transform=plt.gca().transAxes, ha="center", fontsize=12)
        plt.scatter(mountain_data["thickness"], mountain_data["height_abatt"], facecolor="darkslategrey", alpha=0.7)

        plt.xlabel("Layer thickness (m)", fontsize=12)
        if row == 3:
            plt.ylabel("Elevation (m above datum)", fontsize=12)

        plt.subplot(len(mountain_names), 3, 3 + (3 * (row - 1)))
        plt.scatter(mountain_data["width"], mountain_data["thickness"], facecolor="darkslategrey", alpha=0.7)

        plt.xlabel("Layer width (m)", fontsize=12)
        if row == 3:
            plt.ylabel("Layer thickness (m)", fontsize=12)
        
        

    plt.show()

def plot_width_thickness():
    data = prepare_data()

    plt.figure(figsize=(7,8))
    ax0 = plt.subplot(3,1,1)
    ax1 = plt.subplot(3,1,2)
    ax2 = plt.subplot(3,1,3)

    labels = {"height_abatt": "Height above datum (m)", "thickness": "Thickness (m)", "width": "Width (m)"}
    axis_parameters = [
        [ax0, "width", "height_abatt"],
        [ax1, "thickness", "height_abatt"],
        [ax2, "width", "thickness"]
    ]
    marker_shapes = {True: "o", False: "X"}

    labels_table = {"height_abatt": "Height above datum", "thickness": "Thickness", "width": "Width"}    
    statistics = pd.DataFrame(columns=["r-value (all)", "r-value (inliers)", "RMSE-value (all) (m)", "RMSE-value (inliers) (m)"])

    for axis, x_column, y_column in axis_parameters:

        model = RANSACRegressor()
        model.fit(data[x_column].values.reshape(-1, 1), data[y_column])
        linear_width_values = np.linspace(data[x_column].min(), data[x_column].max())
        predicted_thicknesses = model.predict(linear_width_values.reshape(-1, 1))
        data["inlier"] = model.inlier_mask_
        residuals = data[y_column] - model.predict(data[x_column].values.reshape(-1, 1))
        rmse_all = np.sqrt(np.mean(np.square(residuals)))
        rmse_inliers = np.sqrt(np.mean(np.square(residuals[model.inlier_mask_])))
        R_coeff_all = np.corrcoef(data[y_column], model.predict(data[x_column].values.reshape(-1, 1)))[1,0]
        R_coeff_inliers = np.corrcoef(data[y_column][model.inlier_mask_], model.predict(data[x_column].values.reshape(-1, 1))[model.inlier_mask_])[1,0]
        
        statistics.loc[f"{labels_table[x_column]} vs {labels_table[y_column]}"] = R_coeff_all, R_coeff_inliers, rmse_all, rmse_inliers
        

        for i, (mountain_name, mountain_data) in enumerate(data.groupby("mountain_name")):
            nice_label = mountain_name.replace("oe","ø").replace("_"," ")
            color = plt.get_cmap("twilight")(i / len(np.unique(data["mountain_name"])))
            
            for inlier, marker_style in marker_shapes.items():
                in_or_outlier_data = mountain_data[mountain_data["inlier"] == inlier]
                axis.scatter(in_or_outlier_data[x_column],in_or_outlier_data[y_column],label=nice_label if inlier else None, c=(color,), edgecolors="darkgrey", marker=marker_style)
        
        axis.set_xlim(data[x_column].min(), data[x_column].max())
        sns.regplot(data=data[model.inlier_mask_], x=x_column, y=y_column, scatter=False, ci=90, ax=axis, truncate=False)
        axis.set_xlabel(labels[x_column])
        axis.set_ylabel(labels[y_column])
        axis.text(0.5, 0.9, s=f"y = {model.estimator_.coef_[0]:.3f}x + {model.estimator_.intercept_:.3f}", transform=axis.transAxes, ha="center",)

    statistics.round(3).to_csv("statistics_single_plot.csv")
    ax1.legend()
    plt.tight_layout()
    plt.show()

def plot_bins():
    data = prepare_data()
    step = 50

    bins = np.arange(data["height_abatt"].min() - data["height_abatt"].min() % step, data["height_abatt"].max() + step, step=step)
    data["bin"] = np.digitize(data["height_abatt"],bins=bins)
    data["outlier"] = False
    whis = 1.5

    fig = plt.figure(figsize=(8, 5))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    for bin, bin_data in data.groupby("bin"):
        mean_height = np.mean([bins[bin - 1], bins[bin]])

        Q3_thickness = np.quantile(bin_data["thickness"], 0.75)
        Q1_thickness = np.quantile(bin_data["thickness"], 0.25)
        thickness_outlier = Q3_thickness + whis * (Q3_thickness - Q1_thickness)

        Q3_width = np.quantile(bin_data["width"], 0.75)
        Q1_width = np.quantile(bin_data["width"], 0.25)
        width_outlier = Q3_width + whis * (Q3_width - Q1_width)

        data.loc[(data["bin"]==bin) & ((data["thickness"] > thickness_outlier) | (data["width"] > width_outlier)), "outlier"] = True

        boxplot_params = dict(
            vert=False,
            positions=[mean_height],
            widths=[step * 0.9],
            boxprops=dict(
                alpha=1,
                facecolor="white"
            ),
            patch_artist=True,
        )

        box1 = ax1.boxplot([bin_data["thickness"]], **boxplot_params)
        box2 = ax2.boxplot([bin_data["width"]], **boxplot_params)
    
    ax3.scatter(data.loc[data["outlier"],"width"], data.loc[data["outlier"],"thickness"], color="white", edgecolors="black", zorder=2)
    ax3.scatter(data.loc[~data["outlier"],"width"], data.loc[~data["outlier"],"thickness"], edgecolors="black", zorder=2)

    mean_heights = np.mean([bins[:-1], bins[1:]], axis=0)
    height_labels = [f"{bins[i]:.0f}⁠–{bins[i + 1]:.0f}" for i in range(mean_heights.shape[0])]
    ax1.set_yticks(mean_heights)
    ax1.set_yticklabels(height_labels)
    ax2.set_yticks(mean_heights)
    ax2.set_yticklabels([])

    ax2.tick_params(axis="y", direction="in")
    ax1.grid(axis="y")
    ax2.grid(axis="y")
    ax1.set_xlim(-3, 26)
    ax1.set_ylabel("Height interval (m above datum)")
    ax1.set_xlabel("Thickness (m)")
    ax2.set_xlabel("Width (m)", labelpad=1)
    ax3.set_ylabel("Thickness (m)")
    ax3.set_xlabel("Width (m)", labelpad=1)
    ax3.grid(zorder=1)
    ax2.set_xscale("log")
    ax3.set_xscale("log")

    ax1.text(0.025, 0.91, "A)", transform=ax1.transAxes, fontsize=12)
    ax2.text(0.025, 0.91, "B)", transform=ax2.transAxes, fontsize=12)
    ax3.text(0.01, 0.91, "C)", transform=ax3.transAxes, fontsize=12)

    model = LinearRegression(fit_intercept=False)
    model.fit(data["width"].values.reshape(-1, 1), data["thickness"].values.reshape(-1, 1))
    modelled_thickness = model.predict(data["width"].values.reshape((-1, 1)))

    r_value = np.corrcoef(data["thickness"].values, modelled_thickness.squeeze())[0, 1]
    print(r_value)
    print(data)

    
    plt.subplots_adjust(left=0.114, bottom=0.09, right=0.997, top=0.97, wspace=0.01, hspace=0.243)
    plt.savefig("figures/layer_statistics.jpg", dpi=600)
    plt.show()

def estimate_datum_height():
    boundary_points = pd.read_csv("batt_boundary_redrawn.csv")
    boundary_points.rename(columns={"Height_asl": "Height"}, inplace=True)
    planes = {}
    for locality, locality_data in boundary_points.groupby("Locality"):
        model = estimate_plane(locality_data)
        planes[locality] = model
    return planes

if __name__ == "__main__":
    #plot_width_thickness()
    #print(prepare_data())
    #plot_data()
    plot_bins()
    #for filename in os.listdir("layer_boundaries/"):
        #if not filename.endswith(".csv"):
            #continue
        #boundary_height.grid_points_new(os.path.join("layer_boundaries/", filename), f"layer_boundary_rasters/{filename.replace('.csv', '.tif')}")
    #boundary_height.grid_points_new("boundary_height.csv", "boundary_height_new.tif")
    #boundary_height.boundary_surface()
    #estimate_datum_height()