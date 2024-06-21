# Multi-Thresholding Method for Levee Detection

This Python code implements the multi-thresholding method proposed by Sasaki et al. in 2023 for identifying levees in detailed elevation maps. The original paper can be accessed [here](https://doi.org/10.3178/hrl.17.9).

## Method Overview

- The method employs a combination of threshold values for elevation, slope, aspect, and curvature to identify levees.
- Each pixel is evaluated against these thresholds, and a pixel is marked as part of a levee if it consistently meets the criteria across all parameters.
- The process involves multiple steps:
  - Calculating aspect difference and slope from the elevation data.
  - Computing profile curvature to capture topographical features.
  - Determining relative elevation within a defined kernel size.
  - Applying thresholding techniques to filter out relevant features.
  - Combining the thresholded arrays to generate a final levee index map.

## Requirements

- Python 3
- Required Python packages: `rasterio`, `numpy`, `scipy`, `matplotlib`, `gdal` (for GeoTIFF creation)

## Usage

1. Ensure you have the necessary Python packages installed.
2. Provide the input elevation map in the specified format (`LiDAR_DEM` variable).
3. Run the code step by step or as a complete script to generate the levee index map.
4. Adjust threshold values as needed for your specific dataset by modifying the corresponding code sections.

## Code Structure

- Import necessary libraries (`rasterio`, `numpy`, etc.).
- Load the elevation data and preprocess it (remove values smaller than a threshold).
- Calculate aspect difference, slope, profile curvature, and relative elevation.
- Apply thresholding techniques to each parameter individually.
- Combine thresholded arrays to compute the levee index map.
- Optionally, create a GeoTIFF output for the levee index map.

## Example Outputs

Three example levee index maps are provided in the code comments, showcasing different thresholding configurations.

## Author

- **M. Jaweed Nazary**
- University of Missouri - Columbia
- March 2024

For detailed implementation and methodology, refer to the inline comments in the code.
