import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_dataset(filepath: str, interpolation_step: float = 1) -> pd.DataFrame:
    """
    Reads dataset and interpolates equally spaced points.

    :param filepath: Relative path to file.
    :param interpolation_step: Optional. The interval at which to interpolate points.
    :returns: Interpolated dataset without original points.
    """
    dataset = pd.read_csv(filepath, delimiter="\t", header=None, names=["Easting", "Northing", "Height"])
    # Dataset is added a last point which equals the first data point to close the shape. This equals the polygon's circumference.
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
    plt.plot(dataset["Northing"], dataset["Height"])
    plt.show()

def estimate_plane(dataset: pd.DataFrame):
    return

if __name__ == "__main__":
    dataset = read_dataset(r"Data\Polygons_Litledalsfjellet\SS_39.asc")
    print(dataset)
    plot_from_xdir(dataset)
