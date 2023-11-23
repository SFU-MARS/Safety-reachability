import cv2
import os
from pathlib import Path

image_folder = './traj_imgs_area3/1/'
video_name = 'video_area3_1.avi'

images = [img for img in os.listdir(image_folder) if img.startswith("test")]
images = sorted(
    images,
    key=lambda x: '_'.join([f'{int(float(i)):06d}' for i in Path(x).stem.split('_') if i.split('.')[0].isdigit()])
)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
