from flask import *
from flask_socketio import SocketIO, emit
from flask_uploads import UploadSet
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from io import StringIO
import io
from PIL import Image
import base64

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

cam = cv2.VideoCapture(0)


def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
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


def gen_frames(data_image):
    num_frames = 0
    # sbuf = StringIO()
    # sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    while True:
        # success, frame = cam.read()
        success, frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            # ROI from the frame
            roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (15, 15), 0)
            if num_frames < 70:

                cal_accum_avg(gray_frame, accumulated_weight)

                cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
                            (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            else:
                # segmenting the hand region
                hand = segment_hand(gray_frame)

                # Checking if we are able to detect the hand...
                if hand is not None:

                    thresholded, hand_segment = hand
                    # Drawing contours around hand segment
                    cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
                                                                  ROI_top)], -1, (255, 0, 0), 1)

                    cv2.imshow("Thesholded Hand Image", thresholded)

                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(thresholded,
                                               cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(thresholded,
                                             (1, thresholded.shape[0], thresholded.shape[1], 3))
                    # thresholded = np.expand_dims(thresholded, axis=0)
                    pred = model.predict(thresholded)
                    cv2.putText(frame_copy, str(np.argmax(pred[0])),
                                (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw ROI on frame_copy
            cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,
                                                            ROI_bottom), (255, 128, 0), 3)
            # incrementing the number of frames for tracking
            num_frames += 1
            # Display the frame with segmented hand
            cv2.putText(frame_copy, "hand sign recognition",
                        (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
            ret, buffer = cv2.imencode('.jpg', frame_copy)
            frame = buffer.tobytes()
            print(str(np.argmax(pred[0])))
            emit('text', {'image': str(np.argmax(pred[0]))})
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


media = UploadSet('media', extensions=('mp4'))
# @socketio.on('image', namespace='/video_feed')


@app.route('/video_feed', methods=['POST'])
def video_feed():
    # raw_data = request.get_data()
    if "video" in request.files:
        video = request.files["video"]
        # filename = secure_filename(file.filename) # Secure the filename to prevent some kinds of attack
        media.save(video, name="uploaded_file")
    return Response("Great Success", mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True, host="localhost")
