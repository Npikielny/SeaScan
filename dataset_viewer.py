import cv2, tqdm, glob, tyro
from pathlib import Path

def main(root_path: str):
    images = glob.glob(str(
        Path(root_path) / "**" / "*.JPG"
        ))
    
    if len(images) == 0:
        images = glob.glob(str(
            Path(root_path)  / "*.JPG"
            ))
    

    for f in tqdm.tqdm(sorted(images)):
        cv2.imshow('frame', cv2.imread(f))
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)