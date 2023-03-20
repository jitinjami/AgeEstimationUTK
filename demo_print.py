import argparse
import better_exceptions
from pathlib import Path
from contextlib import contextmanager
import urllib.request
import numpy as np
import cv2
import dlib
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from model import get_model
from defaults import _C as cfg

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

def yield_images_from_dir(img_dir):
    img_dir = Path(img_dir)
    img = cv2.imread(str(img_dir), 1)

    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        yield cv2.resize(img, (int(w * r), int(h * r))), img_dir.name

def main():
    '''
    This script reads an image, predicts age from a model, and prints on console
    '''
    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = Path(__file__).resolve().parent.joinpath("checkpoint","epoch000_0.02855_8.2543.pth")#"misc", "epoch044_0.02343_3.9984.pth")
    checkpoint = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])

    if device == "cuda":
        cudnn.benchmark = True

    model.eval()
    margin = 0.4
    img_dir = Path(__file__).resolve().parent.joinpath("photo1.jpg")
    detector = dlib.get_frontal_face_detector()
    img_size = cfg.MODEL.IMG_SIZE
    image_generator = yield_images_from_dir(img_dir)

    with torch.no_grad():
        for img, name in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
                outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs * ages).sum(axis=-1)
                print("Estimated age is: {}".format(int(predicted_ages)))

if __name__ == '__main__':
    main()