# SeaScan

This software leverages targets of opportunity at the sea surface to make maps without the use of expensive structure-from-motion that often fails on the sea surface. [For a more in-depth explanation please visit our project page!](https://npikielny.github.io/SeaScan)

## File Structure
To run our software, please organize all of your images in a folder structure as such:

data_root/
├── folder1/
│   ├── image_1.JPG
│   │ ... 
│   └── image_n.JPG
├── folder2/
│   ├── image_1.JPG
│   │ ...
│   └── image_n.JPG

The names of the folders and images are unimportant, as long as the images are JPGs with the capitalized extension. 

As you use our software, it will ask to specify a working directory (it will create it if it does not exist). In this folder, it will write various log files and will store its data under a `project.npy`. If you create products, they will be stored in this folder

## Automatic Target Detection
To run the automatic detection script, run the following:
python preproc.py --target {INSERT PATH TO TARGET IMAGE} --target_area {INSERT AREA SIZE} --working_directory {INSERT WORKING DIRECTORY}  --data {INSERT DATA ROOT} --calibration {INSERT CALIBRTION PATH}

Parameters:
target: an image that has a target (ideally this is in the center of an area with **many** targets–at least 3)
target_area: the radius (in degree arc seconds) around the target to search for targets
working_directory: the working directory path
data: the root directory of your data
calibration (optional): path to the calibration.npy file for the data

## Manual Target Detection
To run the manual detection server, run the following:
python manual_triangulation.py --working {WORKING DIRECTORY} --root /Volumes/BroadKey0313/DCIM/26_04_01 --calibration ./calibration.npy

In the terminal window, after some processing, there will be a local link. Paste this into your browser.

## Folium Map Generation

## Geotiff Generation

## Timeseries Generation
