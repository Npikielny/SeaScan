# SeaScan

Constructing maps of flat regions should be a relatively simple task, but conventional approaches require complicated structure-from-motion pipelines intended for full 3D reconstruction.
These methods often fail on scenes without dinstiguishable details, such as the ocean. We overcome this limitation through the use of target(s) and GPS data to stitch maps that are focused at a singular depth.

> [!Abstract] Map of Waters Near the Rosenstiel School using the MAVIC 3E
> <div style="display: flex; justify-content: space-around">
>  <img 
>    src="./figures/Rosenstiel HTML Map May 6.png"
>    alt="drawing"
>    width="300"
>  />
> <video controls src="./figures/rectified_timelapse_small.mp4" width="600"></video>
> </div>

> [!FAILURE] MAYBE MAKE THE ABOVE CALLOUT A TLAPSE?

> [!Note] Accessibility
> Our method was designed to automatically detect mooring buoys for boats; however, manually isolating a target in just a few images yields similar results. Thus, our method can be used on any type of target with minimal effort. 
> <div style="display: flex; justify-content: space-around">
>  <img 
>    src="./figures/may_6_features.jpg"
>    alt="drawing"
>    width="350"
>  />
>  <img 
>    src="./figures/demo_manual.png"
>    alt="drawing"
>    width="350"
>  />
> </div>

## Products

### Folium Map
The simplest result is a map made by folium
<iframe src="./figures/march_20_folium.html" width="100%" height="500px"></iframe>

### Stitched Image


### Timelapse Videos




## Introduction
Mapping software often creates a 3D model of its target before making maps. From this new representation, it can reproject the geometry into new coordinates, such as latitude and longitude. 

To create this 3D model, it estimates where cameras are and what way the cameras are facing (pose estimation). This allows it to fill in tiny details and estimate the 3D structure itself. This process requires the software to find the same objects across multiple images; however, this is not always possible on the ocean. Many parts of the sea surface look exactly like other parts. In these conditions, camera pose estimation often fails. Thus, software like Agisoft Metashape only works for maps in shallow areas.

> [!FAILURE] ADD FOCUSING PART

> [!FAILURE] Metashape Results Over Deep Water
> <div style="display: grid; place-items: center;">
>  <img 
>    src="./figures/Rosenstiel HTML Map May 6.png"
>    alt="drawing"
>    width="500"
>  />
> </div>
> Here is a mesh the ACES team made using an Astro Drone at the Rosenstiel pier. Notice the images in the bottom left corner with no corresponding results. These images were not successfully connected to their surrounding cameras, so the software didn't know how to align the images.

In this project, we leverage constraints unique to sea-surface mapping to simplify and solve this problem through GPS coordinates and heading information stored in the MAVIC 3E's metadata. With our method, we can generate maps of much larger and deeper swaths in **tens of minutes**.

## Methods
A drone moving laterally whilst facing nadir at a plane will cause a lateral translation in its images. This translation is a function of the distance between the drone and that plane. While altitude is often recorded in the metadata of images, it is a complicated measurement that does not necessarily correspond to the distance important here. There could be multiple sources of error, but most importantly, tidal shifts mean that any measurement relative to mean sea level is insufficient. Additionally, the intrinsics of the camera are often not included in the metadata and are important for this function. Moreover, camera intrinsics may change depending on settings, so it may be impractical to measure beforehand.

Thus, we must estimate this transformation per flight. **This assumes the drone will not fly long enough for a substantial tidal shift.** To do so, on the sea surface, we find shared points across multiple images. We use these targets to find the pixel shifts between images. Then, we can find a linear mapping between GPS shifts and pixel shifts. Conveniently, there are often buoys that fulfill this purpose; however, they can introduce noise if they move significantly.

![[ransac_match_325.jpg]]
We can automatically detect these buoys by their bright color and shape. We can then match buoys by matching the image around them. Large glint areas can greatly decrease the number of buoys found and will limit the matches found. The program requires at least 3 matches to consider pairing good enough for matching. 
This could also be done manually. This would require fewer images and only one shared point in each image.




