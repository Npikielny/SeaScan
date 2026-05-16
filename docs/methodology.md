
## Introduction
Mapping software often creates a 3D model of its target before making maps. From this new representation, it can reproject the geometry into new coordinates, such as latitude and longitude. 

To create this 3D model, SFM pipelines estimate where cameras are and what way the cameras are facing (pose estimation). This allows enables the software to fill in tiny details and estimate the 3D structure itself. This process requires the same objects to be captured and found in multiple images; however, this is not always possible on the ocean. Many parts of the sea surface look exactly like other parts. In these conditions, camera pose estimation often fails. Thus, software like Agisoft Metashape only works for maps in shallow areas.

> [!CAUTION] ❌ Metashape Results Over Deep Water 
> <div style="display: grid; place-items: center;">
>  <img 
>    src="./figures/Metashape_Error.png"
>    alt="drawing"
>    width="500"
>  />
> </div>
> Here is a mesh the ACES team made using an Astro Drone at the Rosenstiel pier. Notice the images in the bottom left corner with no corresponding results (the cameras are respresented with spheres instead of rectangles). These images were not successfully connected to their surrounding cameras, so the software didn't know how to align them.

In this project, we leverage constraints unique to sea-surface mapping to simplify and solve this problem through GPS coordinates and heading information stored in the MAVIC 3E's metadata. With our method, we can generate maps of much larger and deeper swaths in **tens of minutes**.

## Theory
Structure from motion uses parallax. Imagine you're in a car; objects closer to the car will appear to move faster than objects farther away. From this, one can ascertain the object's distance to the car. One can do the same with objects in a photo. Say you have two photos of the same area from a shifted perspective (assume the cameras are pointing the same way, they just moved horizontally), you may be able to shift the images around to align them. But, due to parallax, only objects at a certain distance from the camera will be aligned. Normally, this complicates the problem; however, when mapping the sea surface, everything is at the same distance! If we can find a mapping between camera translation and image translation, we can stitch images together. These stitched images will have the interesting property that only objects at the sea surface will be in "focus"–objects above or below the water will blur.

To estimate this transformation, we find targets on the sea surface in a few images and then run an optimization to estimate this transformation.


## Methods
While altitude is often recorded in the metadata of images, it is a complicated measurement that does not necessarily correspond to the important distance here. There could be multiple sources of error, but most importantly, tidal shifts mean that any measurement relative to mean sea level would be inaccurate. Additionally, the intrinsics of the camera are often not included in the metadata and are important for this function. Moreover, camera intrinsics may change depending on settings, so it may be impractical to measure beforehand.

Thus, we must estimate this transformation per flight. **This assumes the drone will not fly long enough for a substantial tidal shifts.** To do so, on the sea surface, we find shared points across multiple images. We use these targets to find the pixel shifts between images. Then, we can find a linear mapping between GPS shifts and pixel shifts. Conveniently, there are often buoys that fulfill this purpose; however, they can introduce noise if they move significantly, so it may be best to prioritize buoys in calm locations.

![RANSAC Example](./figures/ransac_match_325.jpg)
We can automatically detect these buoys by their bright color and shape. We can then match buoys based upon the contents of the image around them. Large glint areas can greatly decrease the number of buoys found and will limit the matches found. The program requires at least 3 matches to consider pairing good enough for matching. Due to the simple nature of the targets, we create a simple, specialized feature detector and descriptor. Since the matching is simpler than a traditional counterpart like SIFT, we use RANSAC to ensure a consistent image translation. Images that do not have sufficient matches after RANSAC are ignored.

This could also be done manually, which requires fewer images and only one shared point in each image. Thus, we allow for any target of opportunity to be used. 

After the translations between multiple images are found, the algorithm uses RANSAC to estimate the transformation between GPS coordinates and image translations.




