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
`python preproc.py --target {INSERT PATH TO TARGET IMAGE} --target_area {INSERT AREA SIZE} --working_directory {INSERT WORKING DIRECTORY}  --data {INSERT DATA ROOT} --calibration {INSERT CALIBRTION PATH}`

Parameters:
target: an image that has a target (ideally this is in the center of an area with **many** targets–at least 3)
target_area: the radius (in degree arc seconds) around the target to search for targets
working_directory: the working directory path
data: the root directory of your data
calibration (optional): path to the calibration.npy file for the data

You will find summaries of the results in the working folder. In the `features` directory, you will find all the areas the algorithm considered as a feature. In the `matches` folder you will find the final matchings between images that were used to estimate the transformation. 

## Manual Target Detection
To run the manual detection server, run the following:
`python manual_triangulation.py --working {WORKING DIRECTORY} --root {INSERT DATA ROOT} --calibration ./calibration.npy`

In the terminal window, after some processing, there will be a local link. Paste this into your browser. It will usually be: [http://127.0.0.1:7860](http://127.0.0.1:7860), but it may change if you have multiple instances running or if this port is already being used.

At this point, you will see the following:

On the left is a plot of the coordinates of your data. The green point is the designated target image. The surrounding red images are the other images that will be used to estimate the transform. After pressing the "Update" button, you can view these images in the panel on the right.
Please set the latitude, longitude, and target area size (Dist) to encompass the images with one target shared among them. **You must use the same target every time**. Press finish when you're happy with your selection.

Then you will be prompted to find the target within each image. On the left you will see a reference image. 
Each image will be presented twice. The top image allows you to zoom in. Click twice on opposite corners of your target to make a box–the software will remember only your last two clicks. Then, in the bottom pane, there will be a zoomed in portion of the image from the box you made. In this section, click on the center of your target. If you're happy with your selection, press save. If you cannot find the target, press skip.

On the left side, a reference will be shown. Please find the target in this pane as well and save it.

If the images fail to load, you may havee to press the `skip` button and then the `back` button, to refresh.

When you are done with all of the images (skipped or saved), press the finish button.

## Folium Map Generation

## Geotiff Generation

## Timeseries Generation
