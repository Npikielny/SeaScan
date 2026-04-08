import glob
import folium
import rasterio
from rasterio.plot import reshape_as_image
from folium.raster_layers import ImageOverlay
from tqdm import tqdm
results = glob.glob("./*.tif")

images = []
bounds = []
for r in results:
    print("Loading", r)
    with rasterio.open(r) as src:
        img = reshape_as_image(src.read())  # converts (bands, H, W) -> (H, W, bands)
        images.append(img)
        bounds.append(src.bounds)
        print(src.bounds)

print("Creating map")
# Create map centered on raster
m = folium.Map(location=[(bounds[0].top+bounds[0].bottom)/2,
                         (bounds[0].left+bounds[0].right)/2], zoom_start=22)

# folium.TileLayer('Stamen Terrain').add_to(m)
# folium.TileLayer('Stamen Toner').add_to(m)
# folium.TileLayer('Stamen Watercolor').add_to(m)

# folium.LayerControl().add_to(m)
folium.TileLayer(
    tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
    attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
    name='Stamen Terrain'
).add_to(m)

for (im, b) in tqdm(zip(images, bounds), "Adding Tiles"):
    # Add GeoTIFF as image overlay
    ImageOverlay(
        image=im,
        bounds=[[b.bottom, b.left], [b.top, b.right]],
        opacity=0.7,
    ).add_to(m)

print("Saving Map")
m.save("map.html")