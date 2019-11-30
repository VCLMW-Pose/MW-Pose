#-*- coding = utf-8 -*-
"""
# Copyright (c) 2018-2019, Shrowshoo-Young All Right Reserved.
#
# This programme is developed as free software for the investigation of human
# pose estimation using RF signals. Redistribution or modification of it is
# allowed under the terms of the GNU General Public Licence published by the
# Free Software Foundation, either version 3 or later version.
#
# Redistribution and use in source or executable programme, with or without
# modification is permitted provided that the following conditions are met:
#
# 1. Redistribution in the form of source code with the copyright notice
#    above, the conditions and following disclaimer retained.
#
# 2. Redistribution in the form of executable programme must reproduce the
#    copyright notice, conditions and following disclaimer in the
#    documentation and\or other literal materials provided in the distribution.
#
# This is an unoptimized software designed to meet the requirements of the
# processing pipeline. No further technical support is guaranteed.
"""

##################################################################################
#                                   capture.py
#
#   Definitions of Walabot capture script.
#
#   Shrowshoo-Young 2019-11, shaoshuyangseu@gmail.com
#   Southeast University, Vision and Cognition Laboratory, 211189 Nanjing, China
##################################################################################

from pynput.keyboard import Controller, Key, Listener
from matplotlib import animation

import progressbar
import WalabotAPI
import threading
import time
import sys
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np

# Walabot raw image
walabot_raw_image = 0
walabot_image_slice = 0
walabot_heatmap = 0

# Walabot instance
walabot = WalabotAPI
camera = cv2.VideoCapture(1)

# Shutter for capture, 0 if shutter not pressed. 1 shutter pressed and now capturing.
capture_shutter = False
capture_end = False
capture_figure = 0
capture_axis = 0
capture_animation = 0
capture_id = 0

# Captured image saving directory
capture_savedir = 'F:/captureNov15'

# Walabot configuration constants
WALABOT_MIN_R = 10
WALABOT_MAX_R = 300
WALABOT_RES_R = 5

WALABOT_MIN_THETA = -45
WALABOT_MAX_THETA = 45
WALABOT_RES_THETA = 3

WALABOT_MIN_PHI = -45
WALABOT_MAX_PHI = 45
WALABOT_RES_PHI = 3

WALABOT_THRESHOLD = 15
WALABOT_EN_MTI = False

# Define continuously capture number
CAPTURE_FRAMES = 10

sensor_size = np.array([(WALABOT_MAX_THETA - WALABOT_MIN_THETA) / WALABOT_RES_THETA + 1,
                        (WALABOT_MAX_PHI - WALABOT_MIN_PHI) / WALABOT_RES_PHI + 1,
                        (WALABOT_MAX_R - WALABOT_MIN_R) / WALABOT_RES_R + 1])


def start_listenting():
    with Listener(on_press=on_shutter_pressed, on_release=on_shutter_released) as listener:
        print('[capture info] Now start listening keyboard input.\n')
        listener.join()

def on_shutter_pressed(key):
    global capture_shutter
    # Shutter is pre-set to SPACE key on the keyboard, press SPACE to start
    # continuous capturing. When capture shutter is still on, the capturing process
    # is running and incapable to launch another capture sequence.
    if key == Key.space and capture_shutter == False:
        print("[capture info] Shutter pressed, now capture %d frames of images\n"
              % CAPTURE_FRAMES)
        capture_shutter = True


def on_shutter_released(key):
    global walabot, capture_shutter
    # Stop capturing when user pressed ESC. This program will not be terminated by ESC
    # before any capturing process ended.
    if key == Key.esc and capture_shutter == False:
        print("[capture info] ESC pressed, capture exited!\n")
        # Disconnection walabot
        walabot.Stop()
        walabot.Disconnect()

        # Close animation and exit this program
        plt.close()
        sys.exit(1)


def walabot_get_slice(raw_image):
    #
    w, h, n = raw_image.shape
    slice = np.zeros([w, n])

    for i in range(w):
        for j in range(n):
            slice[i, j] = np.max(raw_image[i, :, j])

    return slice


def continuous_capture(capture_frames, current_id):
    # Capture progressbar
    widgets = [
        'Capturing: ',
        progressbar.Bar(), ' ',
        progressbar.Counter(), ' ',
        progressbar.Timer(), ' ',
        progressbar.ETA()
    ]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=capture_frames)
    bar.start()

    raw_images = []
    opt_images = []
    tdiff = 0
    # Continuous capture images for capture_frames frames.
    for i in range(capture_frames):
        # Trigger walabot and start scanning
        walabot.Trigger()

        # Provides tridimensional (3-D) image data and 2D optical image.
        raw_image, x, y, z, power = walabot.GetRawImage()
        sensor_time = time.time()
        ret, img = camera.read()
        camera_time = time.time()

        # Measure delay
        tdiff += camera_time - sensor_time

        raw_image = np.array(walabot_raw_image)
        raw_images.append(raw_image)
        opt_images.append(img)
        bar.update(i)

    bar.finish()

    # Average delay
    tdiff /= capture_frames
    print("[capture info] Capture completed, average delay: %f\n", tdiff)

    # Save images
    for i, (sensor, image) in enumerate(zip(raw_images, opt_images)):
        # Write images captured by web camera
        image_path = os.path.join(capture_savedir, '%04d.jpg' % i)
        cv2.imwrite(image_path, image)

        # Write raw sensor image in binary form
        with open(os.path.join(capture_savedir, '%04d' % i), 'w') as file:
            sensor_size.astype(np.int32).tofile(file)
            sensor.astype(np.int32).tofile(file)
            file.close()


def animation_init():
    return walabot_heatmap,


def animation_update(image):
    global capture_shutter, capture_id
    global walabot_raw_image, walabot_image_slice, walabot_heatmap
    # If shutter pressed, turn on continuously capture mode.
    if capture_shutter == True:
        continuous_capture(CAPTURE_FRAMES, capture_id)
        capture_id += CAPTURE_FRAMES
        capture_shutter = False
    else:
        # Trigger walabot and start scanning
        walabot.Trigger()

        # Provides tridimensional (3-D) image data.
        walabot_raw_image, x, y, z, power = walabot.GetRawImage()
        walabot_raw_image = np.array(walabot_raw_image)

        # Provides bidimensional (2-D) image data (3D image is projected to 2D plane) of
        # the slice where the strongest signal is produced.
        walabot_image_slice = walabot_get_slice(walabot_raw_image)
        walabot_image_slice = walabot_image_slice.transpose([1, 0])
        walabot_heatmap = capture_axis.pcolormesh(walabot_image_slice, cmap='jet')


    return walabot_heatmap,

def run_capture():
    global walabot
    global capture_figure, capture_axis, capture_animation
    global walabot_raw_image, walabot_image_slice, walabot_heatmap
    # Initialize walabot API
    walabot.Init()
    walabot.SetSettingsFolder()

    print('[capture info] Walabot initilized.\n')

    # Check connection availability and configure capture settings. They are sensor
    # range R, resolution (in cm) of range arena settings theta and phi and
    # resolution (in deg) of arena, sensitivity of sensor depicted using threshold.
    walabot.ConnectAny()
    walabot.SetProfile(walabot.PROF_SENSOR)
    walabot.SetArenaR(WALABOT_MIN_R, WALABOT_MAX_R, WALABOT_RES_R)
    walabot.SetArenaTheta(WALABOT_MIN_THETA, WALABOT_MAX_THETA, WALABOT_RES_THETA)
    walabot.SetArenaPhi(WALABOT_MIN_PHI, WALABOT_MAX_PHI, WALABOT_RES_PHI)
    walabot.SetThreshold(WALABOT_THRESHOLD)

    print('[capture info] Walabot connection availability confirmed.\n')
    print('[capture info] Walabot configuration completed, ready to engage.\n')

    # Enable Moving Target Identification (MTI) filter if required
    if WALABOT_EN_MTI:
        walabot.SetDynamicImageFilter(walabot.FILTER_TYPE_MTI)

    # Start capture
    walabot.Start()
    walabot.StartCalibration()
    walabot.Trigger()

    # Provides tridimensional (3-D) image data.
    walabot_raw_image, x, y, z, power = walabot.GetRawImage()
    walabot_raw_image = np.array(walabot_raw_image)

    # Sketch raw walabot image
    capture_figure = plt.figure()
    capture_axis = capture_figure.add_subplot(111)

    # Provides bidimensional (2-D) image data (3D image is projected to 2D plane) of
    # the slice where the strongest signal is produced.
    walabot_image_slice = walabot_get_slice(walabot_raw_image)
    walabot_image_slice = walabot_image_slice.transpose([1, 0])
    walabot_heatmap = capture_axis.pcolormesh(walabot_image_slice, cmap='jet')

    # Plot colorbar corresponding to heatmap of 2D walabot image slice
    capture_figure.colorbar(walabot_heatmap)

    # Plot 2D walabot image slice animation
    capture_animation = animation.FuncAnimation(capture_figure,
                                                animation_update,
                                                init_func=animation_init,
                                                repeat=False, interval=0, blit=True)
    plt.show()


def run_capture_test():
    # Initialize walabot API
    walabot.Init()
    walabot.SetSettingsFolder()

    print('[capture info] Walabot initilized.\n')

    # Check connection availability and configure capture settings. They are sensor
    # range R, resolution (in cm) of range arena settings theta and phi and
    # resolution (in deg) of arena, sensitivity of sensor depicted using threshold.
    walabot.ConnectAny()
    walabot.SetProfile(walabot.PROF_SENSOR)
    walabot.SetArenaR(WALABOT_MIN_R, WALABOT_MAX_R, WALABOT_RES_R)
    walabot.SetArenaTheta(WALABOT_MIN_THETA, WALABOT_MAX_THETA, WALABOT_RES_THETA)
    walabot.SetArenaPhi(WALABOT_MIN_PHI, WALABOT_MAX_PHI, WALABOT_RES_PHI)
    walabot.SetThreshold(WALABOT_THRESHOLD)

    print('[capture info] Walabot connection availability confirmed.\n')
    print('[capture info] Walabot configuration completed, ready to engage.\n')

    # Enable Moving Target Identification (MTI) filter if required
    if WALABOT_EN_MTI:
        walabot.SetDynamicImageFilter(walabot.FILTER_TYPE_MTI)

    # Start capture
    walabot.Start()
    walabot.StartCalibration()
    walabot.Trigger()

    # Provides tridimensional (3-D) image data and 2D optical image.
    raw_image, x, y, z, power = walabot.GetRawImage()
    sensor_time = time.time()
    ret, img = camera.read()
    camera_time = time.time()

    # Capture progressbar
    widgets = [
        'Capturing: ',
        progressbar.Bar(), ' ',
        progressbar.Counter(), ' ',
        progressbar.Timer(), ' ',
        progressbar.ETA()
    ]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=CAPTURE_FRAMES)
    bar.start()

    raw_images = []
    opt_images = []
    tdiff = 0
    # Continuous capture images for capture_frames frames.
    for i in range(CAPTURE_FRAMES):
        # Trigger walabot and start scanning
        walabot.Trigger()

        # Provides tridimensional (3-D) image data and 2D optical image.
        raw_image, x, y, z, power = walabot.GetRawImage()
        sensor_time = time.clock()
        ret, img = camera.read()
        camera_time = time.clock()

        # Measure delay
        tdiff += camera_time - sensor_time

        raw_image = np.array(raw_image)
        raw_images.append(raw_image)
        opt_images.append(img)
        bar.update(i)

    bar.finish()

    # Average delay
    tdiff /= CAPTURE_FRAMES
    print("[capture info] Capture completed, average delay: %f\n" % tdiff)

    # Save images
    for i, (sensor, image) in enumerate(zip(raw_images, opt_images)):
        # Write images captured by web camera
        image_path = os.path.join(capture_savedir, '%04d.jpg' % i)
        cv2.imwrite(image_path, image)

        # Write raw sensor image in binary form
        with open(os.path.join(capture_savedir, '%04d' % i), 'w') as file:
            sensor_size.astype(np.int32).tofile(file)
            sensor.astype(np.int32).tofile(file)
            file.close()


if __name__ == "__main__":
    # Star listen keyboard interruptions
    # keyboard_thread = threading.Thread(target=start_listenting)
    # keyboard_thread.start()
    # run_capture()

    # For debugging purpose
    run_capture_test()