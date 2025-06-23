import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models

model = models.load_model('hand_gesture_model.h5')

class HandDetector:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # Hand landmark names for reference
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        # Key landmarks for fingertips and important points
        self.key_landmarks = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20,
            'index_mcp': 5,
            'middle_mcp': 9,
            'ring_mcp': 13,
            'pinky_mcp': 17
        }
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
    def detect_hands(self, frame):
        """Detect hands and return landmark positions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_data = []
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand classification (Left/Right)
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                
                # Extract landmark positions
                landmarks = []
                key_points = {}
                
                h, w, c = frame.shape
                
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z  # Depth (relative to wrist)
                    
                    landmarks.append({
                        'id': idx,
                        'name': self.landmark_names[idx],
                        'x': x,
                        'y': y,
                        'z': z,
                        'normalized_x': landmark.x,
                        'normalized_y': landmark.y
                    })
                    
                    # Store key landmarks
                    for key_name, key_id in self.key_landmarks.items():
                        if idx == key_id:
                            key_points[key_name] = (x, y)
                
                hand_data.append({
                    'hand_label': hand_label,
                    'landmarks': landmarks,
                    'key_points': key_points,
                    'raw_landmarks': hand_landmarks
                })
        
        return hand_data, results
    
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on the frame"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw_styles.get_default_hand_landmarks_style(),
                    self.mp_draw_styles.get_default_hand_connections_style()
                )
    
    def draw_key_points(self, frame, hand_data):
        """Draw key points with labels"""
        colors = {
            'wrist': (255, 0, 0),      # Blue
            'thumb_tip': (0, 255, 0),   # Green
            'index_tip': (0, 0, 255),   # Red
            'middle_tip': (255, 255, 0), # Cyan
            'ring_tip': (255, 0, 255),   # Magenta
            'pinky_tip': (0, 255, 255)   # Yellow
        }
        
        for hand in hand_data:
            hand_label = hand['hand_label']
            key_points = hand['key_points']
            
            # Draw fingertips with different colors
            for point_name, (x, y) in key_points.items():
                if point_name in colors:
                    color = colors[point_name]
                    cv2.circle(frame, (x, y), 8, color, -1)
                    cv2.circle(frame, (x, y), 12, color, 2)
                    
                    # Add label
                    label = f"{point_name.upper()}"
                    cv2.putText(frame, label, (x + 15, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_info(self, frame, hand_data):
        """Draw hand information on frame"""
        y_offset = 30
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Draw number of hands detected
        cv2.putText(frame, f"Hands detected: {len(hand_data)}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Draw hand information
        for i, hand in enumerate(hand_data):
            hand_label = hand['hand_label']
            cv2.putText(frame, f"Hand {i+1}: {hand_label}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 25
            
            # Show key point coordinates
            key_points = hand['key_points']
            if 'index_tip' in key_points:
                x, y = key_points['index_tip']
                cv2.putText(frame, f"  Index tip: ({x}, {y})",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 20
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_time)
            self.fps_counter = 0
            self.fps_time = current_time
    
    def run_webcam(self,frame):
        """Run hand detection using webcam"""
        #cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # print("Hand Detection Application")
        # print("=" * 30)
        # print("Controls:")
        # print("- Press 'q' to quit")
        # print("- Press 's' to save current hand positions")
        # print("- Press 'p' to print hand positions to console")
        # print("=" * 30)
        
        # saved_positions = []
        
        # while True:
        #     ret, frame = cap.read()
        # if not ret:
        #     print("Error: Could not read frame")
        #     break
        
        # Flip frame horizontally for mirror effect
        #frame = cv2.flip(frame, 1)
        
        # Detect hands
        hand_data, results = self.detect_hands(frame)
        
        # Draw landmarks and key points
        #self.draw_landmarks(frame, results)
        #self.draw_key_points(frame, hand_data)
        self.draw_info(frame, hand_data)
        
        # Display frame
        cv2.imshow('Hand Detection', frame)
        
        # Update FPS
        self.update_fps()
        
        # # Handle keyboard input
        # key = cv2.waitKey(1) & 0xFF
        
        # if key == ord('q'):
        #     break
        # elif key == ord('s'):
        #     if hand_data:
        #         saved_positions.append(hand_data)
        #         print(f"Saved hand positions! Total saved: {len(saved_positions)}")
        #     else:
        #         print("No hands detected to save!")
        # elif key == ord('p'):
        #     if hand_data:
        #         self.print_hand_positions(hand_data)
        #     else:
        #         print("No hands detected!")
        
        # cap.release()
        # cv2.destroyAllWindows()
        
        return hand_data
    
    def print_hand_positions(self, hand_data):
        """Print hand positions to console"""
        print("\n" + "="*50)
        print("CURRENT HAND POSITIONS")
        print("="*50)
        
        for i, hand in enumerate(hand_data):
            print(f"\nHand {i+1} - {hand['hand_label']}:")
            print("-" * 30)
            
            # Print key points
            for point_name, (x, y) in hand['key_points'].items():
                print(f"{point_name:12}: ({x:3d}, {y:3d})")
            
            # Print all landmarks if needed
            print(f"\nAll {len(hand['landmarks'])} landmarks:")
            for landmark in hand['landmarks']:
                print(f"{landmark['name']:20}: ({landmark['x']:3d}, {landmark['y']:3d}, {landmark['z']:6.3f})")
        
        print("="*50)
    
    def detect_from_image(self, image_path):
        """Detect hands from a static image"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        hand_data, results = self.detect_hands(frame)
        
        # Draw landmarks and key points
        self.draw_landmarks(frame, results)
        self.draw_key_points(frame, hand_data)
        
        # Display result
        cv2.imshow('Hand Detection - Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return hand_data
def preprocess_hands_data(saved_positions):
    """Preprocess saved hand data for further analysis or training"""
    processed_data = []
    for hand_data in saved_positions:
        for hand in hand_data:
            landmarks = hand['landmarks']
            points=[]
            #key_points = hand['key_points']
            for i in range(len(landmarks)):
                landmark = landmarks[i]
                points.append((landmark['x'], landmark['y'], landmark['z']))
            
            processed_data.append({
                'hand_label': hand['hand_label'],
                'points': points
            })
    return processed_data
# Example usage
cap = cv2.VideoCapture(0)
saved_positions = []
detector = HandDetector()
print("Hand Detection Application")
print("=" * 30)
print("Controls:")
print("- Press 'q' to quit")
print("- Press 's' to save current hand positions")
print("- Press 'p' to print hand positions to console")
print("=" * 30)
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Flip frame horizontally for mirror effect
    #frame = cv2.flip(frame, 1)
    
    # Run hand detection
    hand_data = detector.run_webcam(frame)
    temp1 = preprocess_hands_data([hand_data])
    temp1 = [hand['points'] for hand in temp1]
    temp1 = np.array(temp1)
    
    if temp1.shape[0] != 0:
        mines = np.min(temp1, axis=1)
        maxes = np.max(temp1, axis=1)
        temp1 = (temp1 - mines[:, np.newaxis]) / (maxes - mines)[:, np.newaxis]
        output = model.predict(temp1,verbose=0)
        output = np.argmax(output, axis=1)
        print(output)

    
    
    # Display the processed frame
    cv2.imshow('Hand Detection', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        saved_positions.append(hand_data)
        print(f"Saved hand positions! Total saved: {len(saved_positions)}")
    elif key == ord('p'):
        if hand_data:
            detector.print_hand_positions(hand_data)
        else:
            print("No hands detected!")

if saved_positions:
    saved_positions = preprocess_hands_data(saved_positions)
    import pickle
    only_points = []
    for hand_data in saved_positions:
            only_points.append(hand_data['points'])
    only_points = np.array(only_points)
    maxes = np.max(only_points,axis=1)
    mines = np.min(only_points,axis=1)
    normalized_points = (only_points - mines[:, np.newaxis]) / (maxes - mines)[:, np.newaxis]

    # Save the processed data to a file
    with open('processed_hand_data_4.pkl', 'wb') as f:
        pickle.dump(normalized_points, f)
# Example: Detect from image (uncomment to use)
# hand_data = detector.detect_from_image('hand_image.jpg')
# if hand_data:
#     detector.print_hand_positions(hand_data)