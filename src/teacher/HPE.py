# -*- coding: utf-8 -*-
'''
    Created on Wed Sep 25 23:32 2018

    Author           : Shaoshu Yang, Yu Du
    Email            : shaoshuyangseu@gmail.com, yuduseu@gmail.com
    Last edit date   : 

South East University Automation College, 211189 Nanjing China
'''

from src.teacher.utils import *
from src.teacher.detect import detector
from src.teacher.model import *
# from demo_hg import *
import torch
import cv2
import time



class HPE():
    def __init__(self):
        # Deploy darknet53 model on cooresponding device
        yolov3 = darknet('D:/Documents/Source/MW-Pose/config/yolov3.cfg', 80)
        yolov3.load_weight("D:/Documents/Source/MW-Pose/config/yolov3.weights")
        yolov3.eval()

        # Deploy stacked hourglass model
        # stackedhourglass = demo_hg.__dict__['hg'](num_stacks=2, num_blocks=1, num_classes=16)
        stackedhourglass = hg(num_stacks=2, num_blocks=1, num_classes=16)
        stackedhourglass = torch.nn.DataParallel(stackedhourglass)
        stackedhourglass.eval()
        checkpoint = torch.load('D:/Documents/Source/MW-Pose/config/hg_s2_b1/model_best.pth.tar')
        stackedhourglass.load_state_dict(checkpoint['state_dict'])

        cuda = torch.cuda.is_available()
        if cuda:
            yolov3.cuda()
            stackedhourglass.cuda()

        self.detector = detector(yolov3)
        self.estimator = stackedhourglass

    # Capture the frount camera
    def video_cap(self):
        cap = cv2.VideoCapture(0)

        while 1:
            ret, frame = cap.read()
            cv2.imshow("capture", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    # Execute human detection on the image sequences from camera
    def human_det(self):
        # Capture the camera
        cap = cv2.VideoCapture(0)

        while 1:
            ret, frame = cap.read()
            try:
                # Making prediction
                prediction = self.detector.detect(frame)

                # Only person class proposals are needed
                pred_bbox = []
                for prediction_ in prediction:
                    if prediction_[6] == 0:
                        pred_bbox.append(prediction_[:4])

                # Drawing bounding-box
                self.draw_bbox(pred_bbox, frame)

                # Press 'q' to exit
                cv2.imshow("target", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            except:
                cv2.imshow("target", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        cap.release()

    def draw_bbox(self, prediction, img):
        '''
            Args:
                 prediction       : (list) list that record the prediction bounding-box
                 img              : (ndarray) original image
            Returns:
                 Image with bounding-box on it
        '''
        for prediction_ in prediction:
            coord1 = tuple(map(int, prediction_[:2]))
            coord2 = tuple(map(int, prediction_[2:4]))
            cv2.rectangle(img, coord1, coord2, (0, 255, 0), 2)

        return img

    def single_pose_estimate(self):
        cap = cv2.VideoCapture(0)

        while 1:
            ret, frame = cap.read()
            # frame = cv2.imread('data/samples/messi.jpg')
            out_img = frame.copy()
            try:
                # Geting dimensions, normalization and transforming
                img_h, img_w = frame.shape[0], frame.shape[1]

                coord = []
                # Crop bounding-box

                # Resize and padding
                new_h = int(img_h * min(256 / img_h, 256 / img_w))
                new_w = int(img_w * min(256 / img_h, 256 / img_w))
                pad_h = (256 - new_h) // 2
                pad_w = (256 - new_w) // 2

                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                canvas = np.full((256, 256, 3), 128)
                canvas[(256 - new_h) // 2:(256 - new_h) // 2 + new_h, (256 - new_w) // 2:(256 - new_w) // 2 + new_w,
                :] = frame

                # Normalization
                canvas = torch.FloatTensor(canvas[:, :, ::-1].transpose((2, 0, 1)).copy()).div(255.).unsqueeze(0)

                # Get output heatmaps
                output = self.estimator(canvas)[1][0]

                for key_point in output:
                    if key_point.max() > 0.05:
                        # Key point coordinate
                        x, y = map(float, (np.where(key_point == key_point.max())))
                        x *= 4
                        y *= 4

                        # Transfer to original scale
                        x -= pad_h
                        y -= pad_w
                        x *= img_h / new_h
                        y *= img_w / new_w

                        coord.append(tuple(map(int, (y, x))))
                        # cv2.circle(out_img, tuple(map(int, (y, x))), 2, (0, 255, 0), 2)
                    else:
                        coord.append((0, 0))

                draw(out_img, coord, 2)

                # Press 'q' to exit
                cv2.imshow("target", out_img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            except:
                cv2.imshow("target", out_img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        cap.release()

    def pose_estimate(self):
        cap = cv2.VideoCapture(0)

        while 1:
            ret, frame = cap.read()
            # frame = cv2.imread('data/samples/messi.jpg')
            out_img = frame.copy()
            try:
                # Geting dimensions, normalization and transforming
                frame_h, frame_w = frame.shape[0], frame.shape[1]

                begin = time.clock()
                # Making prediction
                prediction = self.detector.detect(frame)

                # Only person class proposals are needed
                pred_bbox = []
                for prediction_ in prediction:
                    if prediction_[6] == 0:
                        w = prediction_[2] - prediction_[0]
                        h = prediction_[3] - prediction_[1]
                        prediction_[0] -= 0.2 * w
                        prediction_[1] -= 0.2 * h
                        prediction_[2] += 0.2 * w
                        prediction_[3] += 0.2 * h
                        pred_bbox.append(prediction_[:4])

                # Prepare container for key point coordinates
                estimation = []

                # Get estimation
                for pred_bbox_ in pred_bbox:
                    pred_bbox_ = list(map(int, pred_bbox_))

                    # Coordinates shall not exceed the boundary of origin image
                    for i in range(2):
                        pred_bbox_[2 * i] = pred_bbox_[2 * i] if pred_bbox_[2 * i] >= 0 else 0
                        pred_bbox_[2 * i] = pred_bbox_[2 * i] if pred_bbox_[2 * i] <= frame_w else frame_w

                    for i in range(2):
                        pred_bbox_[2 * i + 1] = pred_bbox_[2 * i + 1] if pred_bbox_[2 * i + 1] >= 0 else 0
                        pred_bbox_[2 * i + 1] = pred_bbox_[2 * i + 1] if pred_bbox_[2 * i + 1] <= frame_w else frame_w

                    # Crop bounding-box
                    img = frame[pred_bbox_[1]:pred_bbox_[3], pred_bbox_[0]:pred_bbox_[2], :]
                    img_h, img_w = img.shape[0], img.shape[1]

                    # Resize and padding
                    new_h = int(img_h * min(256 / img_h, 256 / img_w))
                    new_w = int(img_w * min(256 / img_h, 256 / img_w))
                    pad_h = (256 - new_h) // 2
                    pad_w = (256 - new_w) // 2

                    img_ = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    canvas = np.full((256, 256, 3), 128)
                    canvas[(256 - new_h) // 2:(256 - new_h) // 2 + new_h, (256 - new_w) // 2:(256 - new_w) // 2 + new_w,
                    :] = img_

                    # Normalization
                    canvas = torch.FloatTensor(canvas[:, :, ::-1].transpose((2, 0, 1)).copy()).div(255.).unsqueeze(0)

                    # Get output heatmaps
                    output = self.estimator(canvas)[1][0]

                    coord = []
                    for key_point in output:
                        if key_point.max() > 0.05:
                            # Key point coordinate
                            x, y = map(float, (np.where(key_point == key_point.max())))
                            x *= 4
                            y *= 4

                            # Transfer to original scale
                            x -= pad_h
                            y -= pad_w
                            x *= img_h / new_h
                            y *= img_w / new_w
                            x += pred_bbox_[1]
                            y += pred_bbox_[0]

                            coord.append(tuple(map(int, (y, x))))
                        else:
                            coord.append((0, 0))

                    draw(out_img, coord, 2)

                end = time.clock()
                # Press 'q' to exit
                cv2.imshow("target", out_img)
                print(end - begin)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            except:
                cv2.imshow("target", out_img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

        cap.release()


                    
    def img_annotate(self, frame):

        out_img = frame.copy()

        # Geting dimensions, normalization and transforming
        img_h, img_w = frame.shape[0], frame.shape[1]

        coord = []
        # Crop bounding-box

        # Resize and padding
        new_h = int(img_h * min(256 / img_h, 256 / img_w))
        new_w = int(img_w * min(256 / img_h, 256 / img_w))
        pad_h = (256 - new_h) // 2
        pad_w = (256 - new_w) // 2

        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((256, 256, 3), 128)
        canvas[(256 - new_h) // 2:(256 - new_h) // 2 + new_h, (256 - new_w) // 2:(256 - new_w) // 2 + new_w,
        :] = frame

        # Normalization
        canvas = torch.FloatTensor(canvas[:, :, ::-1].transpose((2, 0, 1)).copy()).div(255.).unsqueeze(0)

        # Get output heatmaps
        output = self.estimator(canvas)[1][0]
        # op_np = np.zeros((16, 2), dtype=int)
        # for part in range(0, 16):
        #     part_output = output[0, part + 16, :, :]
        #     if part_output.max() != 0:  # and part_output.max() >= thresh:
        #         op_np[part][0] = np.where(part_output == part_output.max())[0][0]
        #         op_np[part][1] = np.where(part_output == part_output.max())[1][0]
        output = output.cpu()
        for key_point in output:
            if key_point.max() > 0.05:
                # Key point coordinate
                x, y = map(float, (np.where(key_point == key_point.max())))
                x *= 4
                y *= 4

                # Transfer to original scale
                x -= pad_h
                y -= pad_w
                x *= img_h / new_h
                y *= img_w / new_w

                coord.append(tuple(map(int, (y, x))))
                # cv2.circle(out_img, tuple(map(int, (y, x))), 2, (0, 255, 0), 2)
            else:
                coord.append((0, 0))

        # draw(out_img, coord, 2)

        # cv2.imshow("target", out_img)
        return coord

    def img_annotate_multi(self, frame):
        # Geting dimensions, normalization and transforming
        frame_h, frame_w = frame.shape[0], frame.shape[1]

        begin = time.clock()
        # Making prediction
        prediction = self.detector.detect(frame)

        # Only person class proposals are needed
        pred_bbox = []
        for prediction_ in prediction:
            if prediction_[6] == 0:
                w = prediction_[2] - prediction_[0]
                h = prediction_[3] - prediction_[1]
                prediction_[0] -= 0.2 * w
                prediction_[1] -= 0.2 * h
                prediction_[2] += 0.2 * w
                prediction_[3] += 0.2 * h
                pred_bbox.append(prediction_[:4])

        # Prepare container for key point coordinates
        estimation = []

        # Get estimation
        for pred_bbox_ in pred_bbox:
            pred_bbox_ = list(map(int, pred_bbox_))

            # Coordinates shall not exceed the boundary of origin image
            for i in range(2):
                pred_bbox_[2 * i] = pred_bbox_[2 * i] if pred_bbox_[2 * i] >= 0 else 0
                pred_bbox_[2 * i] = pred_bbox_[2 * i] if pred_bbox_[2 * i] <= frame_w else frame_w

            for i in range(2):
                pred_bbox_[2 * i + 1] = pred_bbox_[2 * i + 1] if pred_bbox_[2 * i + 1] >= 0 else 0
                pred_bbox_[2 * i + 1] = pred_bbox_[2 * i + 1] if pred_bbox_[2 * i + 1] <= frame_w else frame_w

            # Crop bounding-box
            img = frame[pred_bbox_[1]:pred_bbox_[3], pred_bbox_[0]:pred_bbox_[2], :]
            img_h, img_w = img.shape[0], img.shape[1]

            # Resize and padding
            new_h = int(img_h * min(256 / img_h, 256 / img_w))
            new_w = int(img_w * min(256 / img_h, 256 / img_w))
            pad_h = (256 - new_h) // 2
            pad_w = (256 - new_w) // 2

            img_ = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((256, 256, 3), 128)
            canvas[(256 - new_h) // 2:(256 - new_h) // 2 + new_h, (256 - new_w) // 2:(256 - new_w) // 2 + new_w,
            :] = img_

            # Normalization
            canvas = torch.FloatTensor(canvas[:, :, ::-1].transpose((2, 0, 1)).copy()).div(255.).unsqueeze(0)

            # Get output heatmaps
            output = self.estimator(canvas)[1][0]
            output = output.cpu()
            coord = []
            for key_point in output:
                if key_point.max() > 0.05:
                    # Key point coordinate
                    x, y = map(float, (np.where(key_point == key_point.max())))
                    x *= 4
                    y *= 4

                    # Transfer to original scale
                    x -= pad_h
                    y -= pad_w
                    x *= img_h / new_h
                    y *= img_w / new_w
                    x += pred_bbox_[1]
                    y += pred_bbox_[0]

                    coord.append(tuple(map(int, (y, x))))
                else:
                    coord.append((0, 0))
            return coord





if __name__ == '__main__':
    test = HPE()
    test.pose_estimate()
