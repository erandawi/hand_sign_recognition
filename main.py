from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import logging

# Set TensorFlow logging to error only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

# Load the saved model
model_path = 'model/model.keras'
model = tf.keras.models.load_model(model_path)

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Preprocess keypoints to match training preprocessing
def preprocess_keypoints(hand_landmarks):
    keypoints = []
    for landmark in hand_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y])
    # Normalize keypoints
    keypoints = np.array(keypoints)
    keypoints -= keypoints.min(axis=0)
    keypoints /= keypoints.max(axis=0)
    return keypoints.flatten().reshape(1, -1)

# Implementing a moving average for smoothing predictions
class PredictionSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = []

    def smooth(self, prediction):
        self.predictions.append(prediction)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
        return np.mean(self.predictions, axis=0)

smoother = PredictionSmoother(window_size=5)

capture_enabled = False
predicted_class = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/toggle_capture', methods=['POST'])
def toggle_capture():
    global capture_enabled
    capture_enabled = not capture_enabled
    return ('', 204)

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global predicted_class
    # Ensure prediction is JSON serializable
    if predicted_class is not None:
        return jsonify({'prediction': int(predicted_class)})
    else:
        return jsonify({'prediction': None})

def generate_frames():
    global capture_enabled, predicted_class
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if capture_enabled:
                    # Preprocess keypoints for model input
                    keypoints = preprocess_keypoints(hand_landmarks)
                    prediction = model.predict(keypoints)
                    smoothed_prediction = smoother.smooth(prediction)
                    predicted_class = np.argmax(smoothed_prediction, axis=1)[0]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
