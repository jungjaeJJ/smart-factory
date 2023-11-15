#!/usr/bin/env python3
'''Start factory Import '''
import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
import cv2
import numpy as np
import openvino as ov
from iotdemo import FactoryController
from iotdemo import MotionDetector
from iotdemo import ColorDetector
FORCE_STOP = False


def thread_cam1(q):
    '''Camera 1 Thread'''
    flag = True
    #  MotionDetector
    motion1 = MotionDetector()
    motion1.load_preset('resources/motion.cfg', 'default')
    #  HW1 Open "resources/conveyor.mp4" video clip
    # pylint: disable=E1101
    cap = cv2.VideoCapture('resources/factory/conveyor.mp4')
    if not cap.isOpened():
        print("can't open videofile\r\n")
    else:
        print("can open video\r\n")
    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model('resources/openvino.xml')
    if len(model.inputs) != 1:
        core.error('Sample supports only single input topologies')
        return -1
    if len(model.outputs) != 1:
        core.error('Sample supports only single output topologies')
        return -1
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        # Enqueue "VIDEO: Cam1 live", frame info
        q.put(('VIDEO: Cam1 live', frame))
        # Motion detect
        detected = motion1.detect(frame)
        if detected is None:
            continue
        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO: Cam1 detected', detected))
        # abnormal detect
        input_tensor = np.expand_dims(detected, 0)
        if flag is True:
            ppp = ov.preprocess.PrePostProcessor(model)
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400
            ppp.input().preprocess().resize(
                ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR
                )
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            compiled_model = core.compile_model(model, "CPU")
            flag = False
        #  Inference OpenVINO
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        print(f'{probs}')
        #  in queue for moving the actuator 1
        if probs[0] > 0.0:
            print("Not Good item")
            q.put(("PUSH", 1))
        else:
            print("Good item")
    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    '''Camera 2 Thread'''
    #  MotionDetector
    motion2 = MotionDetector()
    motion2.load_preset('resources/motion.cfg', 'default')
    #  ColorDetector
    color2 = ColorDetector()
    color2.load_preset('resources/color.cfg', 'default')
    #  Open "resources/conveyor.mp4" video clip
    # pylint: disable=E1101
    cap = cv2.VideoCapture('resources/factory/conveyor.mp4')
    if not cap.isOpened():
        print("can't open videofile\r\n")
    else:
        print("can open video\r\n")
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        # Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame))
        # Detect motion
        detected = motion2.detect(frame)
        if detected is None:
            continue
        #  Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 dectected', detected))
        #  Detect color
        predict = color2.detect(detected)
        # Compute ratio
        name, ratio = predict[0]
        ratio = ratio * 100
        print(f"{name}: {ratio:.2f}%")
        #  Enqueue to handle actuator 2
        if name == 'blue' and ratio > 0.5:
            q.put(('PUSH', 2))
    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    '''Imshow funtion'''
    # pylint: disable=E1101
    cv2.namedWindow(title)
    if pos:
        # pylint: disable=E1101
        cv2.moveWindow(title, pos[0], pos[1])
    # pylint: disable=E1101
    cv2.imshow(title, frame)


def main():
    '''Main'''
    global FORCE_STOP
    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")
    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()
    # Create a Queue
    q = Queue()
    # Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,))
    t2 = threading.Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()
    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            # pylint: disable=E1101
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
            try:
                # get an item from the queue.
                # You might need to properly handle exceptions.
                # de-queue name and data
                event = q.get_nowait()
            except Empty:
                continue
            name, frame = event
            if name.startswith("VIDEO:"):
                imshow(name[6:], frame)
            elif name == "PUSH":
                ctrl.push_actuator(frame)
            elif name == "DONE":
                FORCE_STOP = True
            q.task_done()
    t1.join()
    t2.join()
    # pylint: disable=E1101
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()