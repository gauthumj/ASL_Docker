from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class ProcessImage(object):
    def __init__(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        model = tf.keras.models.load_model(r"ASLModel.h5")
        background = None
        accumulated_weight = 0.5
        ROI_top = 100
        ROI_bottom = 300
        ROI_right = 150
        ROI_left = 350

    def cal_accum_avg(self, frame, accumulated_weight):
        global background

        if self.background is None:
            background = frame.copy().astype("float")
            return None
        cv2.accumulateWeighted(frame, self.background, accumulated_weight)

    def segment_hand(self, frame, threshold=25):
        global background

        diff = cv2.absdiff(background.astype("uint8"), frame)

        _, thresholded = cv2.threshold(diff, threshold, 255,
                                       cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(
            thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None
        else:
            hand_segment_max_cont = max(contours, key=cv2.contourArea)
            return (thresholded, hand_segment_max_cont)

    def recognise(self, img):
        num_frames = 0
        while True:
            frame = img

            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            # ROI from the frame
            roi = frame[self.ROI_top:self.ROI_bottom,
                        self.ROI_right:self.ROI_left]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)
            if self.num_frames < 70:

                self.cal_accum_avg(gray_frame, self.accumulated_weight)

                cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
                            (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            else:
                # segmenting the hand region
                hand = self.segment_hand(gray_frame)

                # Checking if we are able to detect the hand...
                if hand is not None:

                    thresholded, hand_segment = hand
                    # Drawing contours around hand segment
                    cv2.drawContours(frame_copy, [hand_segment + (self.ROI_right,
                                                                  self.ROI_top)], -1, (255, 0, 0), 1)

                    cv2.imshow("Thesholded Hand Image", thresholded)

                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(thresholded,
                                               cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(thresholded,
                                             (1, thresholded.shape[0], thresholded.shape[1], 3))
                    # thresholded = np.expand_dims(thresholded, axis=0)
                    pred = self.model.predict(thresholded)
                    cv2.putText(frame_copy, str(np.argmax(pred[0])),
                                (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw ROI on frame_copy
            cv2.rectangle(frame_copy, (self.ROI_left, self.ROI_top), (self.ROI_right,
                                                                      self.ROI_bottom), (255, 128, 0), 3)
            # incrementing the number of frames for tracking
            num_frames += 1
            # Display the frame with segmented hand
            cv2.putText(frame_copy, "hand sign recognition",
                        (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
            cv2.imshow("Sign Detection", frame_copy)
            # Close windows with Esc
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            alph = {'1': "a", '2': "b", '3': "c", '4': "d", '5': "e", '6': "f", '7': "g", '8': "h", '9': "i", '10': "j", '11': "k", '12': "l", '13': "m",
                    '14': "n", '15': "o", '16': "p", '17': "q", '18': "r", '19': "s", '20': "t", '21': "u", '22': "v", '23': "w", '24': "x", '25': "y", '26': "z"}
            return alph[str(np.argmax(pred[0]))]
