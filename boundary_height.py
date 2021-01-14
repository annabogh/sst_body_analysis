import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import os
import pickle
from scipy.interpolate import SmoothBivariateSpline
import scipy.interpolate
import scipy.ndimage
import rasterio as rio

def read_boundary_points():
    points = pd.read_csv("boundary_height.csv")
    return points


def grid_points_new(boundary_points_filepath, output_raster_filepath):
    points = pd.read_csv(boundary_points_filepath).dropna().rename(columns={"HeightBatt1": "Height"})
    resolution = 2000
    x_grid = np.arange(points["Easting"].min(), points["Easting"].max(), step=resolution)
    y_grid = np.arange(points["Northing"].min(), points["Northing"].max(), step=resolution)

    eastings, northings = np.meshgrid(x_grid, y_grid)
    interpolated_heights = scipy.interpolate.griddata(points[["Easting", "Northing"]].values, points["Height"], (eastings, northings), method="linear")

    bounds = {"west": x_grid.min(), "east": x_grid.max(), "south": y_grid.min(), "north": y_grid.max()}
    transform = rio.transform.from_bounds(**bounds, width=interpolated_heights.shape[1], height=interpolated_heights.shape[0])

    interpolated_heights[np.isnan(interpolated_heights)] = -9999

    with rio.open(output_raster_filepath, mode="w", driver="GTiff", width=interpolated_heights.shape[1], height=interpolated_heights.shape[0], crs=rio.crs.CRS.from_epsg(32633), transform=transform, dtype=interpolated_heights.dtype, count=1, nodata=-9999) as raster:
        raster.write(interpolated_heights[::-1, :], 1)

    plt.imshow(interpolated_heights)
    #plt.show()

def grid_points():
    #cache_file = "cache/gridded_points.csv"
    #if os.path.isfile(cache_file):
       # return pd.read_csv(cache_file)

    points = read_boundary_points()
    resolution = 500
    

    x_grid = np.arange(points["Easting"].min(), points["Easting"].max(), step=resolution)
    y_grid = np.arange(points["Northing"].min(), points["Northing"].max(), step=resolution)

    points["x_bin"] = np.digitize(points["Easting"], x_grid)
    points["y_bin"] = np.digitize(points["Northing"], y_grid)

    cache_file = "cache/heights.pkl"

    if os.path.isfile(cache_file):
        with open(cache_file, "rb") as infile:
            heights = pickle.load(infile)
    else:
        heights = np.zeros(shape=(y_grid.shape[0], x_grid.shape[0]), dtype=np.float64) + np.nan
        progress_bar = tqdm(total=x_grid.shape[0] * y_grid.shape[0], desc="Gridding data")
        for x_bin in range(x_grid.shape[0]):
            for y_bin in range(y_grid.shape[0]):
                height = points.loc[(points["x_bin"] == x_bin) & (points["y_bin"] == y_bin), "HeightBatt1"].median()
                heights[y_bin, x_bin] = height
                progress_bar.update()

        progress_bar.close()
        with open(cache_file, "wb") as outfile:
            pickle.dump(heights, outfile)

    eastings, northings = np.meshgrid(x_grid + resolution / 2, y_grid + resolution / 2)
    gridded_data = pd.DataFrame(
        data=np.transpose([eastings.flatten(), northings.flatten(), heights.flatten()]),
        columns=["Easting", "Northing", "Height"]
    ).dropna()
    x_grid_highres = np.arange(points["Easting"].min(), points["Easting"].max(), step=20)
    y_grid_highres = np.arange(points["Northing"].min(), points["Northing"].max(), step=20)
    eastings_highres, northings_highres = np.meshgrid(x_grid_highres + 20 / 2, y_grid_highres + 20 / 2)
    interpolated_heights = scipy.interpolate.griddata(gridded_data[["Easting", "Northing"]].values, gridded_data["Height"], (eastings_highres, northings_highres), method="linear")

    bounds = {"west": x_grid_highres.min(), "east": x_grid_highres.max(), "south": y_grid_highres.min(), "north": y_grid_highres.max()}
    transform = rio.transform.from_bounds(**bounds, width=interpolated_heights.shape[1], height=interpolated_heights.shape[0])

    interpolated_heights[np.isnan(interpolated_heights)] = -9999

    with rio.open("boundary_height.tif", mode="w", driver="GTiff", width=interpolated_heights.shape[1], height=interpolated_heights.shape[0], crs=rio.crs.CRS.from_epsg(32633), transform=transform, dtype=interpolated_heights.dtype, count=1, nodata=-9999) as raster:
        raster.write(interpolated_heights[::-1, :], 1)

    plt.imshow(interpolated_heights, extent=(x_grid.min(), x_grid.max(), y_grid.max(), y_grid.min()), cmap="viridis")
    plt.colorbar()
    plt.scatter(gridded_data["Easting"], gridded_data["Northing"])
    plt.ylim(gridded_data["Northing"].min(), gridded_data["Northing"].max())
    #plt.show()

    
    return gridded_data




def boundary_surface():
    points = grid_points()


    return
    model = make_pipeline(PolynomialFeatures(degree=50),LinearRegression())
    #model = RANSACRegressor(polynomial_model, residual_threshold=3 * np.std(points["Height"]), random_state=0)
    model.fit(points[["Easting", "Northing"]], points["Height"])
    points["Height_estimated"] = model.predict(points[["Easting", "Northing"]])
    print(points)
    plt.scatter(points["Height_estimated"], points["Height"])
    min_height = points["Height"].min()
    max_height = points["Height"].max()
    plt.plot([min_height, max_height], [min_height, max_height])
    rmse = np.sqrt(np.mean(np.square((points["Height"] - points["Height_estimated"])).values))
    print(rmse)
    plt.show()