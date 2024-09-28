from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import json

# Initialize the gesture recognizer
class SignLanguageRecognizer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.model = load_model('sign_language_model.h5')
        self.max_length = 90
        self.sequence = deque(maxlen=self.max_length)
        self.gesture_labels = ['Alive', 'Bad', 'Beautiful', 'Big large', 'Blind', 'Cheap', 'Clean', 'Cold', 'Cool', 'Curved', 
                               'Dead', 'Deaf', 'Deep', 'Dirty', 'Dry', 'Expensive', 'Famous', 'Fast', 'Female', 'Flat', 'Good', 
                               'Happy', 'Hard', 'Healthy', 'Heavy', 'High', 'Hot', 'Light', 'Long', 'Loose', 'Loud', 'Low', 
                               'Male', 'Mean', 'Narrow', 'New', 'Nice', 'Old', 'Poor', 'Quiet', 'Rich', 'Sad', 'Shallow', 
                               'Short', 'Sick', 'Slow', 'Small little', 'Soft', 'Strong', 'Tall', 'Thick', 'Thin', 'Tight', 
                               'Ugly', 'Warm', 'Weak', 'Wet', 'Wide', 'Young']
        self.predictions = []

    def extract_landmarks(self, frame):
        # Extract landmarks from the image frame
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lh = np.zeros(21*3)
        rh = np.zeros(21*3)
        pose = np.zeros(33*4)
        face = np.zeros(468*3)
        
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        if results.face_landmarks:
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
        
        return np.concatenate([lh, rh, pose, face])

    def recognize_gesture(self, landmarks):
        # Process sequence of landmarks and predict gesture
        if len(self.sequence) == self.max_length:
            res = self.model.predict(np.expand_dims(np.array(self.sequence), axis=0))[0]
            self.predictions.append(np.argmax(res))
            if len(self.predictions) > 5:
                self.predictions = self.predictions[-5:]
            if len(self.predictions) >= 3 and len(set(self.predictions[-3:])) == 1:
                return self.gesture_labels[self.predictions[-1]]
        return ""

    def process_frame(self, frame):
        landmarks = self.extract_landmarks(frame)
        self.sequence.append(landmarks)
        return self.recognize_gesture(landmarks)

# Initialize recognizer
recognizer = SignLanguageRecognizer()

def index(request):
    # Render the HTML template for the index page
    labels = recognizer.gesture_labels
    return render(request, 'recognition/index.html', {'labels': labels})

@csrf_exempt
def process_frame(request):
    if request.method == 'POST':
        # Check if 'frame' is included in the request
        print("The request is : ",request)
        if 'frame' not in request.FILES:
            return JsonResponse({'error': 'No frame part'}, status=400)

        frame_file = request.FILES.get('frame')
        if not frame_file:
            return JsonResponse({'error': 'No selected file'}, status=400)

        try:
            # Convert file to an OpenCV image
            image_bytes = frame_file.read()
            image = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # Process the frame with the gesture recognizer
            gesture = recognizer.process_frame(frame)
            return JsonResponse({'gesture': gesture})
        
        except Exception as e:
            # Return error if something goes wrong in processing
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)
