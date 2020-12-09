import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
    """
    Plots polygon from x-direction in northing/height graph.
    """
    plt.plot(dataset["Northing"], dataset["Height"])
    plt.show()



def plot_plane_and_layer(dataset: pd.DataFrame, model: LinearRegression):  
    """
    Calculates estimated Z-values for plane and plots plane and polygon (AKA layer) in 3D view.
    """  
    x_values = [dataset["Easting"].min(), dataset["Easting"].max()] * 2
    y_values = [dataset["Northing"].max()] * 2 + [dataset["Northing"].min()] * 2
    #Calculates predicted z values and makes variable.
    predicted_zvalues = model.predict(X=np.transpose([x_values, y_values]))
    ax = plt.axes(projection="3d")

    ax.plot3D(dataset["Easting"], dataset["Northing"], dataset["Height"])
    print(predicted_zvalues.shape)
    ax.plot_surface(np.array(x_values).reshape((2, 2)), np.array(y_values).reshape((2, 2)), predicted_zvalues.reshape((2, 2)), alpha=0.5)
    plt.show()

def estimate_plane(dataset: pd.DataFrame):
    """
    Makes flat plane based on polygon points.
    """
    model = LinearRegression()
    model.fit(X=dataset[["Easting", "Northing"]], y=dataset["Height"])
    plot_plane_and_layer(dataset, model)

if __name__ == "__main__":
    dataset = read_dataset(r"Data\Polygons_Litledalsfjellet\SS_39.asc")
    print(dataset)
    #plot_from_xdir(dataset)
    estimate_plane(dataset)
