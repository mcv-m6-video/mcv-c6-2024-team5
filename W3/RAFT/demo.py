import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import time


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)

    # convert gray to 3 channels
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_viz_image(img, flo, save_path, viz=False):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    # BGR to RGB
    flo = flo[:, :, [2, 1, 0]]

    print(f"Saving visualization to {save_path}...")
    cv2.imwrite(save_path, flo)
    if viz:

        img_flo = np.concatenate([img, flo], axis=0)

        # import matplotlib.pyplot as plt
        # plt.imshow(img_flo / 255.0)
        # plt.show()

        cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
        cv2.waitKey()

def save_flow_to_image(u, v, valid, filename):
    assert u.shape == v.shape == valid.shape, "Mismatch in dimension of flow components and valid mask"
    
    # Initialize the flow image with all invalid pixels
    flow_img = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint16)
    
    # Set the flow values, scaling and encoding as per KITTI format
    flow_img[..., 0] = (u * 64.0 + 2**15).astype(np.uint16)
    flow_img[..., 1] = (v * 64.0 + 2**15).astype(np.uint16)
    flow_img[..., 2] = valid
    
    flow_img_bgr = np.stack((flow_img[:, :, 2], flow_img[:, :, 1], flow_img[:, :, 0]), axis=-1)
    print(f"Saving raw flow to {filename}...")
    # Save the image
    cv2.imwrite(filename, flow_img_bgr)

VIZ = True
def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cuda')))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            start_time = time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (time.time() - start_time, image1.shape[0], image1.shape[1], image1.shape[2]))

            # Undo padding
            flow_up = padder.unpad(flow_up)
            image1 = padder.unpad(image1)
            # image2 = padder.unpad(image2)

            flo = flow_up[0].permute(1,2,0).cpu().numpy()

            # get the name of the file of image2
            if '/' in imfile2:
                name = imfile2.split('/')[-1]
            else:
                name = imfile2.split('\\')[-1]

            save_flow_to_image(flo[:,:,0], flo[:,:,1], np.ones_like(flo[:,:,0]), f"./W2_video_of/of_to_{name}")

            if VIZ:
                save_viz_image(image1, flow_up, f"./W2_video_of_viz/viz_of_to_{name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
