import glob
import folium
import rasterio
from rasterio.plot import reshape_as_image
from folium.raster_layers import ImageOverlay
from tqdm import tqdm
results = glob.glob("./*.tif")

import numpy as np
from rasterio.transform import xy

images = []
bounds = []
for r in results:
    print("Loading", r)
    with rasterio.open(r) as src:
        img = reshape_as_image(src.read())  # converts (bands, H, W) -> (H, W, bands)
        images.append(img)
        bounds.append(src.bounds)
        print(src.bounds)

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import reshape_as_image
import folium
from folium.raster_layers import ImageOverlay
from tqdm import tqdm

images = []
bounds = []

for r in results:
    print("Processing", r)
    with rasterio.open(r) as src:
        print("SKEWED RASTER")
        # Check if the raster is skewed
        if src.transform.b != 0 or src.transform.d != 0:
            print("Raster is rotated/skewed. Reprojecting to north-up...")
            
            # Compute transform for axis-aligned version
            transform, width, height = calculate_default_transform(
                src.crs, src.crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': src.crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # Create a temporary in-memory raster
            # reprojected_raster = np.zeros((src.count, height, width), dtype=src.dtypes[0])
            # reprojected_raster.fill(src.nodata)
            reprojected_raster = np.full((src.count, height, width), 0, dtype=src.dtypes[0])
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=reprojected_raster[i-1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata
                )
            img = reshape_as_image(reprojected_raster)
            Y, X = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
            ctr = np.array(img.shape[:2])[::-1] / 2
            UV = (np.dstack([Y, X]) - ctr) / ctr
            D = np.sqrt(np.vecdot(UV, UV))
            
            alpha = np.where(np.all(reprojected_raster == 0, axis=0), 0, 1)
            alpha2 = D / D.max()# < 0.33
            alpha2 = 1 - alpha2
            alpha2 *= alpha2
            alpha2 *= alpha2

            # import matplotlib.pyplot as plt
            # plt.imshow(alpha2); plt.colorbar(); plt.show(); 
            
            MAX_ALPHA = 255
            # alpha = np.clip((alpha * alpha2 * MAX_ALPHA), 0, 255).astype(np.uint8)
            alpha = np.clip(alpha * alpha2 * MAX_ALPHA, 0, 255).astype(np.uint8)
            # plt.imshow(alpha); plt.colorbar(); plt.show()
                #.astype(np.uint8)
            if img.shape[2] == 3:
                img = np.dstack([img, alpha])

            bounds.append([
                transform.f + height * transform.e,  # south
                transform.c,                          # west
                transform.f,                           # north
                transform.c + width * transform.a      # east
            ])
        else:
            print("Raster is north-up.")
            img = reshape_as_image(src.read())
            bounds.append([src.bounds.bottom, src.bounds.left, src.bounds.top, src.bounds.right])

        images.append(img)

# Create Folium map centered on first raster
m = folium.Map(location=[(bounds[0][2] + bounds[0][0]) / 2,
                         (bounds[0][3] + bounds[0][1]) / 2], zoom_start=24, max_zoom=30)

# Add basemap
folium.TileLayer(
    tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
    attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
    name='Stamen Terrain'
).add_to(m)

# Add rasters as overlays
for im, b in tqdm(zip(images, bounds), "Adding Tiles", total=len(images)):
    ImageOverlay(
        image=im,
        bounds=[[b[0], b[1]], [b[2], b[3]]],
        opacity=0.7,
    ).add_to(m)

m.save("map.html")
print("Map saved as map.html")