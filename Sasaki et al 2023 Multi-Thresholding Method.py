#!/usr/bin/env python
# coding: utf-8

# #### This document reproduce Sasaki et al 2023 Multi Thresholding Method for performance review. Original paper can be access here: 
# https://doi.org/10.3178/hrl.17.9

# - This is an innovative approach for spotting levees in detailed elevation maps.
# - The method cleverly uses a mix of threshold values for elevation, slope, aspect, and curvature.
# - It makes decisions about whether a `PIXEL` is a levee based on how consistently it meets these thresholds in various combinations.
# - In essence, a levee is flagged when it checks all the boxes for elevation, slope, aspect, and curvature.
# - here is snapshot of the methodolgy. 
# - M. Jaweed Nazary
# - University of Missouri - Columbia March 2024. 
# 
# ![image.png](attachment:image.png)

# In[90]:


import rasterio
import time


# In[3]:


LiDAR_DEM = "JeffCity_0832.img"
img = rasterio.open(LiDAR_DEM)
array_data = img.read(1)
# Replace values smaller than 0.001 with NaN
array_data[array_data < 0.001] = 0


# ## ASPECT DIFFERENCE

# In[85]:


import numpy as np
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
start_time = time.time()
# Assuming you have a DEM, replace this with your actual DEM data
dem_data = array_data

def calculate_aspect(dem):
    # Calculate gradient using Sobel operator
    gradient_x = sobel(dem, axis=1, mode='reflect')
    gradient_y = sobel(dem, axis=0, mode='reflect')

    # Calculate slope
    slope = np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))

    # Calculate aspect (direction of steepest slope)
    aspect = np.arctan2(-gradient_y, gradient_x)

    # Convert aspect from radians to degrees (0 to 360)
    aspect_degrees = np.degrees(aspect)
    aspect_degrees[aspect_degrees < 0] += 360

    return aspect_degrees, slope

# Calculate aspect for the DEM
aspect_map, slope = calculate_aspect(dem_data)

# Plot the aspect map
plt.figure(figsize=(10, 8))
plt.imshow(aspect_map, cmap='viridis', interpolation='nearest')
plt.title('Aspect Map of DEM')
plt.colorbar(label='Aspect (degrees)')
plt.show()
end_time = time.time()
elapsed_time = end_time-start_time
print(elapsed_time)


# In[86]:


start_time = time.time()

aspect_diff = []
for i in range(1, len(aspect_map)-1):
    row = []
    for j in range(1, len(aspect_map)-1):
        up_down = abs(aspect_map[i-1][j] - aspect_map[i+1][j])
        left_right = abs(aspect_map[i][j+1] - aspect_map[i][j-1])
        upper_left_lower_right = abs(aspect_map[i+1][j+1] - aspect_map[i-1][j-1])
        upper_right_lower_left = abs(aspect_map[i-1][j+1] - aspect_map[i+1][j-1])
        cell_value = 180-min(abs(np.array([up_down,left_right, upper_left_lower_right,upper_right_lower_left])-180))
        row.append(cell_value)
    aspect_diff.append(row)

aspect_diff = np.array(aspect_diff)
# Plot the aspect map
plt.figure(figsize=(10, 8))
plt.imshow(aspect_diff, cmap='viridis', interpolation='nearest')
plt.title('Aspect Map of DEM')
plt.colorbar(label='Aspect (degrees)')
plt.show()
end_time = time.time()
elapsed_time = end_time-start_time
print(elapsed_time)


# ## SLOPE

# In[87]:


start_time = time.time()

Slope = slope*180/np.pi
# Plot the aspect map
plt.figure(figsize=(10, 8))
plt.imshow(Slope, cmap='viridis', interpolation='nearest')
plt.title('Slope Map of DEM')
plt.colorbar(label='Slope (degrees)')
plt.show()
end_time = time.time()
elapsed_time = end_time-start_time
print(elapsed_time)


# ## PROFILE CURVATURE 

# In[88]:


start_time = time.time()

import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

# Assuming you have a DEM, replace this with your actual DEM data
dem_data = array_data

def calculate_profile_curvature(dem):
    # Calculate first derivative using Sobel operator
    gradient_x = convolve(dem, np.array([[-1, 0, 1]]), mode='constant')
    gradient_y = convolve(dem, np.array([[-1], [0], [1]]), mode='constant')

    # Calculate second derivative
    curvature_x = convolve(gradient_x, np.array([[-1, 2, -1]]), mode='constant')
    curvature_y = convolve(gradient_y, np.array([[-1], [2], [-1]]), mode='constant')

    # Calculate profile curvature
    profile_curvature = curvature_x + curvature_y

    return profile_curvature

# Calculate profile curvature for the DEM
profile_curvature_map = calculate_profile_curvature(dem_data)

# Plot the profile curvature map
plt.figure(figsize=(10, 8))
plt.imshow(profile_curvature_map, cmap='viridis', interpolation='nearest', vmin = -2, vmax = 2)
plt.title('Profile Curvature Map of DEM')
plt.colorbar(label='Profile Curvature')
plt.show()
end_time = time.time()
elapsed_time = end_time-start_time
print(elapsed_time)


# ## RELATIVE ELEVATION WITHIN 50X50 Kernel

# In[469]:


start_time = time.time()

def calculate_min_max_values(kernel_size, dem):
    half_kernel = kernel_size // 2
    rows, cols = dem.shape
    min_values = np.zeros((rows, cols))
    max_values = np.zeros((rows, cols))
    
    rel_elv = []
    for i in range(half_kernel, rows - half_kernel):
        row = []
        for j in range(half_kernel, cols - half_kernel):
            # Extract the 100x100 kernel
            kernel = dem[i - half_kernel : i + half_kernel + 1, j - half_kernel : j + half_kernel + 1]

            # Calculate the minimum and maximum values within the kernel
            min_values[i, j] = np.min(kernel)
            max_values[i, j] = np.max(kernel)
            row.append(np.max(kernel) - np.min(kernel))
        rel_elv.append(row)
    rel_elv = np.array(rel_elv)

    return rel_elv

# Choose a kernel size
kernel_size = 30

# Calculate min and max values within the moving kernel
rel_elv = calculate_min_max_values(kernel_size, dem_data)


# Plot the original DEM and the calculated min and max values
plt.figure(figsize=(10, 8))

plt.imshow(rel_elv, cmap='terrain', interpolation='nearest', vmax = 20)
plt.title('Relative Elevation')
plt.colorbar(label='Elevation (ft)')



plt.show()
end_time = time.time()
elapsed_time = end_time-start_time
print(elapsed_time)


# ## Thresholding 
# These threshold can be modifed
# ![image.png](attachment:image.png)

# In[288]:


N =1
for min_thresh in [1,2,3,4,5,6,7]:
    rel_elv_copy = np.array(rel_elv)
    array = rel_elv_copy
    # Define the desired range
    lower_limit = min_thresh
    upper_limit = 20
    # Create a boolean mask for values outside the range
    mask = (array < lower_limit) | (array > upper_limit)

    # Replace values outside the range with zero
    array[mask] = 0
    array[~mask] = 1

    # Save the array with a specific name
    np.save("RE_%s.npy" % N, array)

    # Plot the original DEM and the calculated min and max values
    plt.figure(figsize=(10, 8))

    plt.imshow(array, cmap = "gray", vmax = 1, vmin = 0)
    plt.title('Relative Elevation Thresh = %s' %min_thresh)
    plt.colorbar(label='Elevation (ft)')
    N+=1


# In[390]:


N = 1
for max_slope_thresh in [45,50,55,60,65,70,75,80,85]:
    slope_copy = np.array(Slope)
    array = slope_copy
    # Define the desired range
    upper_limit = max_slope_thresh
    # Create a boolean mask for values outside the range
    mask =  (array > upper_limit)

    # Replace values outside the range with zero
    array[~mask] = 0
    array[mask] = 1

    # Save the array with a specific name
    np.save("S_%s.npy" % N, array)

    # Plot the original DEM and the calculated min and max values
    plt.figure(figsize=(10, 8))

    plt.imshow(array, cmap = "gray", vmax = 1, vmin = 0)
    plt.title('Slope Thresh = %s' %max_slope_thresh)
    plt.colorbar(label='Elevation (ft)')
    N+=1


# In[435]:


N = 1
for n in [10,15,20,25,30,35,40,45]:
    aspect_copy = np.array(aspect_diff)
    array = aspect_copy
    # Define the desired range
    lower_limit = 180 - n
    upper_limit = 180 + n
    # Create a boolean mask for values outside the range
    mask = (array < lower_limit) | (array > upper_limit)

    # Replace values outside the range with zero
    array[mask] = 0
    array[~mask] = 1

    # Save the array with a specific name
    np.save("A_%s.npy" % N, array)

    # Plot the original DEM and the calculated min and max values
    plt.figure(figsize=(10, 8))

    plt.imshow(array, cmap = "gray", vmax = 1, vmin = 0)
    plt.title('Aspect Difference Thresh = %s' %n)
    plt.colorbar(label='Elevation (ft)')
    N+=1


# In[436]:


N = 1
for thresh in [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6]:
    
    curvature_copy = np.array(profile_curvature_map)
    array = curvature_copy
    # Define the desired range
    lower_limit = thresh
    upper_limit = 100
    # Create a boolean mask for values outside the range
    mask = (array < lower_limit) | (array > upper_limit)

    # Replace values outside the range with zero
    array[mask] = 0
    array[~mask] = 1
    
    # Save the array with a specific name
    np.save("PC_%s.npy" % N, array)


    # Plot the original DEM and the calculated min and max values
    plt.figure(figsize=(10, 8))

    plt.imshow(array, cmap = "gray", vmax = 1, vmin = 0)
    plt.title('Curvature Thresh = %s' %thresh)
    plt.colorbar(label='Elevation (ft)')
    N+=1


# In[437]:


# Later, you can load the array back into memory
RE_1 = np.load('RE_1.npy')
RE_2 = np.load('RE_2.npy')
RE_3 = np.load('RE_3.npy')
RE_4 = np.load('RE_4.npy')
RE_5 = np.load('RE_5.npy')
RE_6 = np.load('RE_6.npy')
RE_7 = np.load('RE_7.npy')

S_1 = np.load('S_1.npy')
S_2 = np.load('S_2.npy')
S_3 = np.load('S_3.npy')
S_4 = np.load('S_4.npy')
S_5 = np.load('S_5.npy')
S_6 = np.load('S_6.npy')
S_7 = np.load('S_7.npy')
S_8 = np.load('S_7.npy')
S_9 = np.load('S_7.npy')


A_1 = np.load('A_1.npy')
A_2 = np.load('A_2.npy')
A_3 = np.load('A_3.npy')
A_4 = np.load('A_4.npy')
A_5 = np.load('A_5.npy')
A_6 = np.load('A_6.npy')
A_7 = np.load('A_7.npy')
A_8 = np.load('A_8.npy')



PC_1 = np.load('PC_1.npy')
PC_2 = np.load('PC_2.npy')
PC_3 = np.load('PC_3.npy')
PC_4 = np.load('PC_4.npy')
PC_5 = np.load('PC_5.npy')
PC_6 = np.load('PC_6.npy')
PC_7 = np.load('PC_7.npy')
PC_8 = np.load('PC_8.npy')


# In[438]:


RE = np.array([RE_1, RE_2,RE_3,RE_4,RE_5,RE_6,RE_7])
S = np.array([S_1, S_2, S_3, S_4, S_5, S_6, S_7, S_8, S_9])
PC = np.array([PC_1, PC_2, PC_3, PC_4, PC_5, PC_6, PC_7, PC_8])
A = np.array([A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8])


# In[439]:


import numpy as np
array_list = []
for array in RE:
    # Example arrays with different shapes
    larger_array = S_1

    smaller_array = array

    # Get the shapes of the arrays
    larger_shape = larger_array.shape
    smaller_shape = smaller_array.shape

    # Calculate the center position for placing the smaller array in the larger array
    center_position = ((larger_shape[0] - smaller_shape[0]) // 2, (larger_shape[1] - smaller_shape[1]) // 2)

    # Create a new array with the shape of the larger array, filled with zeros
    result_array = np.zeros_like(larger_array)

    # Place the smaller array in the center of the result array
    result_array[center_position[0]:center_position[0] + smaller_shape[0],
                 center_position[1]:center_position[1] + smaller_shape[1]] = smaller_array

    array_list.append(result_array)

RE = np.array(array_list)


# In[440]:


import numpy as np
array_list = []
for array in A:
    # Example arrays with different shapes
    larger_array = S_1

    smaller_array = array

    # Get the shapes of the arrays
    larger_shape = larger_array.shape
    smaller_shape = smaller_array.shape

    # Calculate the center position for placing the smaller array in the larger array
    center_position = ((larger_shape[0] - smaller_shape[0]) // 2, (larger_shape[1] - smaller_shape[1]) // 2)

    # Create a new array with the shape of the larger array, filled with zeros
    result_array = np.zeros_like(larger_array)

    # Place the smaller array in the center of the result array
    result_array[center_position[0]:center_position[0] + smaller_shape[0],
                 center_position[1]:center_position[1] + smaller_shape[1]] = smaller_array

    array_list.append(result_array)

A = np.array(array_list)


# In[441]:


horizontal_dim = S_1.shape[0]
vertical_dim = S_1.shape[1]


# In[442]:


start_time = time.time()


LEVEE_ = np.zeros((horizontal_dim, vertical_dim))
n = 0
for re in RE:
    for s in S: 
        for pc in PC:
            for a in A:
                levee=re*s*pc*a
                LEVEE_ = LEVEE_ + levee
                n+=1

                
end_time = time.time()
elapsed_time = end_time-start_time
print(elapsed_time)                


# In[444]:


n


# In[465]:


plt.figure(figsize=(10, 8))
    
plt.imshow(LEVEE_, cmap = "YlOrRd_r", vmax = 3000, vmin = 0)
plt.title('LEVEE_INDEX')
plt.colorbar(label='Levee')


# In[489]:


levee_copy = np.array(LEVEE_)
array = levee_copy
# Define the desired range
lower_limit = 500
upper_limit = 5000
# Create a boolean mask for values outside the range
mask = (array < lower_limit) | (array > upper_limit)

# Replace values outside the range with zero
array[mask] = 0
array[~mask] = 1

# Save the array with a specific name
np.save("A_%s.npy" % N, array)

# Plot the original DEM and the calculated min and max values
plt.figure(figsize=(10, 8))

plt.imshow(array, cmap = "binary_r")
plt.title('Levee_Locations')


# In[490]:


# Example arrays with different shapes
larger_array = S_1

smaller_array = rel_elv

# Get the shapes of the arrays
larger_shape = larger_array.shape
smaller_shape = smaller_array.shape

# Calculate the center position for placing the smaller array in the larger array
center_position = ((larger_shape[0] - smaller_shape[0]) // 2, (larger_shape[1] - smaller_shape[1]) // 2)

# Create a new array with the shape of the larger array, filled with zeros
result_array = np.zeros_like(larger_array)

# Place the smaller array in the center of the result array
result_array[center_position[0]:center_position[0] + smaller_shape[0],
             center_position[1]:center_position[1] + smaller_shape[1]] = smaller_array


Levee_elv = levee_copy * result_array


# In[521]:


zoomed_ = Levee_elv[650:900]
# Plot the original DEM and the calculated min and max values
plt.figure(figsize=(10, 8))

plt.imshow(zoomed_.T[0:100].T, cmap = "rainbow")
plt.title('Levee_Locations')
plt.colorbar(label='Levee Relative Height  (ft)')


# In[494]:


zoomed_ = Levee_elv[0:100]
# Plot the original DEM and the calculated min and max values
plt.figure(figsize=(10, 8))

plt.imshow(zoomed_.T[100:400].T, cmap = "rainbow")
plt.title('Levee_Locations')
plt.colorbar(label='Levee Relative Height  (ft)')


# In[497]:


zoomed_ = Levee_elv[600:800]
# Plot the original DEM and the calculated min and max values
plt.figure(figsize=(10, 8))

plt.imshow(zoomed_.T[700:900].T, cmap = "rainbow")
plt.title('Levee_Locations')
plt.colorbar(label='Levee Relative Height  (ft)')


# In[508]:


from osgeo import gdal
# Define the GeoTIFF driver
driver = gdal.GetDriverByName("GTiff")


# In[509]:


num = array_data.shape[1]


# In[522]:


xmin


# In[524]:


with rasterio.open(LiDAR_DEM) as src:
    # Get pixel size in x and y directions
    pixel_size_x, pixel_size_y = src.transform[0], -src.transform[4]
    print(src.transform)

    # Get the x-coordinate of the upper-left corner (xmin) and y-coordinate of the upper-left corner (ymax)
    xmin, ymax = src.bounds.left, src.bounds.top


# In[528]:


src.bounds.left


# In[533]:


# Create the raster dataset
output_raster_path = "Sasaki_2023_Example.tif"
output_raster = driver.Create(output_raster_path, num, num, 1, gdal.GDT_Float32)

# Define the transformation
transform = (xmin, pixel_size_x, 0, ymax, 0, -pixel_size_x)
output_raster.SetGeoTransform(transform)

# Set the projection (assuming WGS84 here)
output_raster.SetProjection("ESRI:102697")

# Write the elevation data to the raster band
band = output_raster.GetRasterBand(1)
band.WriteArray(Levee_elv)

# Close the raster dataset
output_raster = None


# ## Example 1 

# ![image.png](attachment:image.png)

# ## Example 2

# ![image.png](attachment:image.png)

# ### Example #3
# 

# ![image.png](attachment:image.png)

# In[ ]:




