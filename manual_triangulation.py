import gradio as gr, tyro, matplotlib.pyplot as plt, numpy as np
import gps

from project import Project

from PIL import Image
from loguru import logger
log = logger.debug 

from pathlib import Path
import os

from preproc import *

import matplotlib.pyplot as plt
plt.style.use('dark_background')

from calib import Calibration

class GUI(object):
    def __init__(self, target, project, root, calib):
        self.project = project
        self.calib = calib
        if not target is None:
            project['target'] = target
            project.write()
        self.root = root
        if project['images'] is None:
            if root is None:
                raise ValueError("Attempted running on empty project without specifying the root data directory")
            images = get_good_images(root)

            assert(len(images) > 0)
            log(f"Working with {len(images)} images.")
            project['images'] = images
            project.write()
        else:
            images = project['images']
            log(f"Working with {len(images)} images.")
        self.images = images

        if project['coords'] is None:
            coords = np.array(
                [gps.get_coords(str(image)) for image in tqdm(images, "Getting Image Coordinates")]
            )
            project['coords'] = coords
            project.write()
        else:
            coords = project['coords']
        if len(coords.shape) > 2:
            coords = gps.to_arc_seconds(coords)
        self.coords = coords

        if project['target'] is None:
            self.target = np.mean(self.coords, axis=0)
        else:
            self.target = project['target']

        self.dist = project["dist_threshold"]
        if self.dist is None:
            self.dist = 1

        self.ids = [np.argmin(np.linalg.norm(coords - self.target, axis=-1))]
        self.flipper_idx = 0
        self.select_ids = 0

        self.selector_data = {}
        self.position_data = {}

        self.saved_targets = {}

# MARK: IMAGE SELECTION
    def set_target_location(self):
        fig, ax = plt.subplots(1)
        ax.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            s=1,
            c='b'
        )

        if not self.target is None:
            dist = np.linalg.norm(self.coords - self.target, axis=-1)
            mask = dist < self.dist
            if np.sum(mask) != 0:
                ordering = np.argsort(dist[mask])
                self.ids = np.arange(self.coords.shape[0])[mask][ordering]
                self.flipper_idx = 0
                ax.scatter(
                    self.coords[mask, 0],
                    self.coords[mask, 1],
                    s=1,
                    c='r'
                )

                best = np.argmin(dist)

                ax.scatter(
                    self.coords[best, 0],
                    self.coords[best, 1],
                    s=1,
                    c='green'
                )

                self.target = self.coords[best]
                self.project.write()
            else:
                raise ValueError("No images match the criteria")

        id, img = self.get_flipper_data()
        return [fig, id, img]

    def update_plot(self, lat, lon, dist):
        # self.target = 
        print(lat, lon, self.target)
        self.target = np.array([float(lat), float(lon)])
        self.dist = float(dist)
        return self.set_target_location()

    def get_flipper_data(self):
        return f"{self.flipper_idx}/{self.ids.shape[0]}", self.calib.open(self.images[self.ids[self.flipper_idx]])

    def target_flipper_decrease(self):
        self.flipper_idx -= 1
        if self.flipper_idx < 0:
            self.flipper_idx = self.ids.shape[0] - 1
        return self.get_flipper_data()
    
    def target_flipper_increase(self):
        self.flipper_idx += 1
        if self.flipper_idx >= self.ids.shape[0]:
            self.flipper_idx = 0
        return self.get_flipper_data()

    def target_image_flipper(self, image):
        self.flipper_idx = max(0, min(self.flipper_idx, self.ids.shape[0] - 1))
        with gr.Column() as image_flipper:
            with gr.Row():
                left = gr.Button("<")
                label = gr.Text(
                    label="Index",
                    value=f"{self.flipper_idx}"
                )
                right = gr.Button(">")
            left.click(
                fn=self.target_flipper_decrease,
                outputs=[label, image]
            )

            right.click(
                fn=self.target_flipper_increase,
                outputs=[label, image]
            )

        return image_flipper, label

    def target_input(self):
        with gr.Column(visible=True) as t_input:
            with gr.Row():
                p, idx, img = self.set_target_location()
                plot = gr.Plot(p)

                with gr.Column():
                    image_rep = gr.Image(img)

                    flipper, flipper_label = self.target_image_flipper(image_rep)
            with gr.Row():
                # the_interface = gr.Interface(
                #     fn=self.update_plot,
                #     inputs=[
                latbox = gr.Textbox(
                    label="Lat",
                    value=str(self.target[0]),
                    interactive=True
                )
                lonbox = gr.Textbox(
                    label="Lon",
                    value=str(self.target[1]),
                    interactive=True
                )
                distbox = gr.Textbox(
                    label="Dist",
                    value=self.dist,
                    interactive=True
                )

                update = gr.Button(
                    "Update"
                )
            update.click(
                fn=self.update_plot,
                inputs=[latbox, lonbox, distbox],
                outputs=[plot, flipper_label, image_rep]
            )
                #     ],
                #     outputs=[plot, image_rep],
                #     visible=True
                # )
        return t_input

    def target_finish(self):
        self.flipper_idx = 1
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), self.calib.open(self.images[self.ids[0]]), self.calib.open(self.images[self.ids[1]])
# MARK: TARGET SELECTION

    def get_cropped_image(self, img_data, selector_id):
        bounds = self.selector_data[selector_id]['bounds']
        point = self.selector_data[selector_id]["point"]

        print(bounds)
        print(bounds[0])
        MIN = np.min(bounds, axis=0).astype(int)
        MAX = np.max(bounds, axis=0).astype(int)

        if (MIN == MAX).all():
            crop = img_data.copy()
        else:
            crop = img_data[int(MIN[0]):int(MAX[0]),int(MIN[1]):int(MAX[1])]
        return self.draw_circle(crop, MIN, point), MIN, MAX

    def draw_circle(self, crop, MIN, point):
        if point is None:
            return crop
        
        point = np.array(point) - np.array(MIN)

        x, y = int(point[0]), int(point[1])
        color = (0, 255, 0)

        print(crop.dtype, x, y)

        out = cv2.circle(np.array(crop), (y, x), 10, color, -1)
        return out

    def get_target_images(self, img_data, selector_id):
        coords = np.dstack(np.meshgrid(
            np.arange(img_data.shape[1]),
            np.arange(img_data.shape[0])
        )[::-1])

        crop, MIN, MAX = self.get_cropped_image(img_data, selector_id)
        if (MIN == MAX).all():
            return img_data / 255, crop
        else:
            mask = np.bitwise_and(coords > MIN, coords < MAX)
            mask = np.bitwise_and(mask[..., 0], mask[..., 1])
            return (mask * 0.75 + 0.25)[..., np.newaxis] * img_data / 255, crop
    
    def handle_bounds_change(self, getter, selector_id):
        def fn(evt: gr.SelectData):
            bounds = self.selector_data[selector_id]['bounds']
            i = evt.index[1] 
            j = evt.index[0]
            
            bounds = [bounds[1], [np.round(i), np.round(j)]]
            self.selector_data[selector_id]['bounds'] = bounds

            img_data = np.asarray(getter(self))
            return self.get_target_images(img_data, selector_id)
        
        return fn
    
    def handle_target_set(self, getter, selector_id):
        def fn(evt: gr.SelectData):
            i = evt.index[1]
            j = evt.index[0]

            point = [i, j]
            self.selector_data[selector_id]["point"] = np.array(point) + np.min(np.array(self.selector_data[selector_id]['bounds']), axis=0)

            return self.get_cropped_image(
                np.asarray(getter(self)), 
                selector_id
            )[0]
        return fn
        
    def make_selector(self, getter):
        selector_id = self.select_ids
        self.select_ids += 1

        self.selector_data[selector_id] = {
            "bounds": [[0, 0], [0, 0]],
            "point": [0, 0]
        }

        with gr.Column():
            img_data = getter(self)
            bounds_image = gr.Image(img_data)
            clipped_image = gr.Image(img_data)
            bounds_image.select(
                self.handle_bounds_change(getter, selector_id),
                [],
                [bounds_image, clipped_image]
            )

            clipped_image.select(
                self.handle_target_set(getter, selector_id),
                [],
                clipped_image
            )
        return bounds_image, clipped_image
            
    def save_target(self, image_id, selector_id):
        self.saved_targets[
            image_id
        ] = self.selector_data[selector_id]['point']

    def update_target(self):
        text = f"ID:{self.flipper_idx}, Saved:{len(self.saved_targets)}, Total:{self.ids.shape[0]}"

        res = self.saved_targets.get(self.ids[self.flipper_idx])
        if res is None:
            self.selector_data[1][
                'point'
            ] = [0, 0]
        else:
            self.selector_data[1][
                'point'
            ] = res

        self.selector_data[1][
            'bounds'
        ] = [[0, 0], [0, 0]]

        return *self.get_target_images(np.asarray(self.calib.open(self.images[self.ids[self.flipper_idx]])), 1), text

    def target_back(self):
        self.flipper_idx -= 1
        if self.flipper_idx < 1:
            self.flipper_idx = self.ids.shape[0] - 1
        return self.update_target()
    
    def target_next(self):
        self.flipper_idx += 1
        if self.flipper_idx >= self.ids.shape[0]:
            self.flipper_idx = 1
        return self.update_target()
    
    def save_sample_target(self):
        self.save_target(self.ids[self.flipper_idx], 1)
        return self.target_next()


    def target_selector(self):
        with gr.Column(visible=False) as selector:
            with gr.Row():
                self.flipper_idx = 0
                with gr.Column():
                    image1, clipped1 = self.make_selector(lambda self: self.calib.open(self.images[self.ids[0]]))
                    save1 = gr.Button("Save")
                with gr.Column():
                    image2, clipped2 = self.make_selector(lambda self: self.calib.open(self.images[self.ids[max(0, min(self.flipper_idx, self.ids.shape[0] - 1))]]))
                    with gr.Row():
                        back = gr.Button("Back")
                        skip = gr.Button("Skip")
                        save = gr.Button("Save")
                        logBox = gr.Text(f"ID:{self.flipper_idx}, Saved:{len(self.saved_targets)}, Total:{self.ids.shape[0]}")

                        back.click(
                            fn=self.target_back,
                            outputs=[image2, clipped2, logBox]
                        )

                        skip.click(
                            fn=self.target_next,
                            outputs=[image2, clipped2, logBox]
                        )

                        save.click(
                            fn=self.save_sample_target,
                            outputs=[image2, clipped2, logBox]
                        )

                # selector_target = 
                # selector_id =
            finish = gr.Button("Finish")

            save1.click(
                fn=lambda: self.save_target(self.ids[0], 0),
                inputs = []
            )
        return selector, image1, clipped1, image2, clipped2, finish
    
    def complete(self):
        self.project['target_ids'] = self.ids
        self.project["target"] = self.images[self.ids[0]]
        self.project['targets'] = np.array(self.images)[self.ids]
        self.project['target_data'] = self.saved_targets
        self.project.write()
        return gr.update(visible=False), gr.update(visible=True)

    def get_UI(self):
        with gr.Blocks() as demo:
            image_selection_input = self.target_input()
            image_selection_finish = gr.Button("Finish")

            target_selection, selector1, clipped1, selector2, clipped2, target_finish = self.target_selector()

            ender = gr.Text(
                value="Finished! :)",
                interactive=False,
                visible=False
            )

            image_selection_finish.click(
                fn=self.target_finish,
                inputs=[],
                outputs=[image_selection_input, image_selection_finish, target_selection, selector1, selector2]
            )

            target_finish.click(
                fn=self.complete,
                inputs=[],
                outputs=[target_selection, ender]
            )





            
                    


        return demo

def main(
        working: str, 
        root: str | None = None,
        target: str | None = None, 
        calibration: str | None = None
        ):
    if ".npy" in working:
        project = Project(working)
    else:
        if not Path(working).exists():
            os.mkdir(working)
        project = Project(Path(working) / "project.npy")
    gui = GUI(target, project, root, Calibration(calibration))
    gui.get_UI().launch()

if __name__ == "__main__":
    tyro.cli(main)