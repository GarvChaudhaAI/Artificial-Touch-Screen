import cv2
import numpy as np
import time
import mediapipe as mp
import pyautogui as cursor
from tensorflow.keras import models
import tensorflow as tf

#model = models.load_model('hand_gesture_model.h5')

class PointSelector:
    def __init__(self):
        self.points = []
        self.max_points = 4
        self.frame_width = 800
        self.frame_height = 600
        self.window_name = "Point Selector - Click 4 Points"
        
        # Colors for visualization
        self.point_color = (0, 255, 0)  # Green
        self.line_color = (255, 0, 0)   # Blue
        self.text_color = (255, 255, 255)  # White
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.max_points:
                self.points.append((x, y))
                print(f"Point {len(self.points)}: ({x}, {y})")
                
                if len(self.points) == self.max_points:
                    print("\nAll 4 points selected!")
                    print("Points stored:", self.points)
                    print("Press 'r' to reset, 'q' to quit")
    
    def draw_frame(self):
        """Create and draw the current frame"""
        # Create a black frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        # Draw grid for reference
        grid_size = 50
        for i in range(0, self.frame_width, grid_size):
            cv2.line(frame, (i, 0), (i, self.frame_height), (30, 30, 30), 1)
        for i in range(0, self.frame_height, grid_size):
            cv2.line(frame, (0, i), (self.frame_width, i), (30, 30, 30), 1)
        
        # Draw selected points
        for i, point in enumerate(self.points):
            # Draw point circle
            cv2.circle(frame, point, 8, self.point_color, -1)
            cv2.circle(frame, point, 12, self.point_color, 2)
            
            # Draw point number
            cv2.putText(frame, str(i + 1), 
                       (point[0] - 5, point[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)
        
        # Draw lines connecting points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(frame, self.points[i], self.points[i + 1], self.line_color, 2)
        
        # If we have 4 points, connect the last to the first to close the shape
        if len(self.points) == 4:
            cv2.line(frame, self.points[3], self.points[0], self.line_color, 2)
        
        # Draw instructions
        instructions = [
            f"Points selected: {len(self.points)}/{self.max_points}",
            "Left click to select points",
            "Press 'r' to reset points",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 25
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (self.frame_width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)
        
        # Draw point coordinates
        if self.points:
            y_start = self.frame_height - 120
            cv2.putText(frame, "Selected Points:", 
                       (10, y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
            
            for i, point in enumerate(self.points):
                coord_text = f"P{i+1}: ({point[0]}, {point[1]})"
                cv2.putText(frame, coord_text, 
                           (10, y_start + 20 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.point_color, 1)
        
        return frame
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_time)
            self.fps_counter = 0
            self.fps_time = current_time
    
    def reset_points(self):
        """Reset all selected points"""
        self.points = []
        print("Points reset!")
    
    def run(self):
        """Main application loop"""
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("4-Point Selector Application")
        print("=" * 30)
        print("Instructions:")
        print("- Left click to select up to 4 points")
        print("- Press 'r' to reset points")
        print("- Press 'q' to quit")
        print("=" * 30)
        
        while True:
            # Create and display frame
            frame = self.draw_frame()
            cv2.imshow(self.window_name, frame)
            
            # Update FPS
            #self.update_fps()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_points()
            elif key == 27:  # ESC key
                break
        
        # Cleanup
        cv2.destroyAllWindows()
        
        # Print final results
        if self.points:
            print("\nFinal selected points:")
            for i, point in enumerate(self.points):
                print(f"Point {i+1}: {point}")
        else:
            print("\nNo points were selected.")
        
        return self.points
cap = cv2.VideoCapture(0)  # Initialize webcam capture
ret, frame = cap.read()
# Run the application
selector = PointSelector()
selector.frame_width = frame.shape[1]
selector.frame_height = frame.shape[0]
selected_points = selector.run()
def warp_quadrilateral_to_unit_square(quad_points) -> np.ndarray:
    """
    Compute perspective transformation matrix to warp any quadrilateral to unit square at origin.
    
    Args:
        quad_points: List of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] or numpy array of shape (4,2)
                    Points should be in order: top-left, top-right, bottom-right, bottom-left
    
    Returns:
        3x3 perspective transformation matrix
    """
    # Convert to numpy array if needed
    if isinstance(quad_points, list):
        quad_points = np.array(quad_points, dtype=np.float32)
    else:
        quad_points = quad_points.astype(np.float32)
    
    # Define unit square corners at origin (0,0 to 1,1)
    unit_square = np.array([
        [0, 0],  # top-left
        [1, 0],  # top-right
        [1, 1],  # bottom-right
        [0, 1]   # bottom-left
    ], dtype=np.float32)
    
    # Compute perspective transformation matrix
    # Maps from quadrilateral to unit square
    transform_matrix = cv2.getPerspectiveTransform(quad_points, unit_square)
    
    return transform_matrix
def apply_warp_to_points(points, 
                        transform_matrix) -> np.ndarray:
    """
    Apply perspective transformation to a set of points.
    
    Args:
        points: Points to transform, shape (N, 2)
        transform_matrix: 3x3 perspective transformation matrix
    
    Returns:
        Transformed points, shape (N, 2)
    """
    # Convert to numpy array if needed
    if isinstance(points, list):
        points = np.array(points, dtype=np.float32)
    else:
        points = points.astype(np.float32)
    
    # Add homogeneous coordinate
    points_homo = np.column_stack([points, np.ones(len(points))])
    
    # Apply transformation
    transformed_homo = transform_matrix @ points_homo.T
    
    # Convert back to 2D coordinates
    transformed_points = (transformed_homo[:2] / transformed_homo[2]).T
    
    return transformed_points
transform_matrix = warp_quadrilateral_to_unit_square(selected_points)
selected_points = apply_warp_to_points(selected_points, transform_matrix)
# You can use the selected_points list for further processing
print(f"\nReturned points: {selected_points}")


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
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Detect hands
        hand_data, results = self.detect_hands(frame)
        
        # Draw landmarks and key points
        #self.draw_landmarks(frame, results)
        #self.draw_key_points(frame, hand_data)
        #self.draw_info(frame, hand_data)
        
        # Display frame
        #cv2.imshow('Hand Detection', frame)
        
        self.update_fps()
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
import time
import threading
from queue import Queue, Empty
from collections import deque
import ctypes
from ctypes import wintypes
#  class OptimizedMouseController:
#     def __init__(self, min_interval=0.05):
#         self.min_interval = min_interval
#         self.last_action_time = 0
#         self.scroll_accumulator = 0
#         self.last_scroll_time = 0
#         self.scroll_batch_time = 0.1  # Batch scrolls every 100ms
        
#         # For threading approach
#         self.use_threading = True
#         if self.use_threading:
#             self.action_queue = Queue(maxsize=10)  # Limit queue size
#             self.worker_thread = threading.Thread(target=self._worker, daemon=True)
#             self.worker_thread.start()
    
#     def _worker(self):
#         """Background worker for mouse actions"""
#         while True:
#             try:
#                 action_type, cursor, timestamp = self.action_queue.get(timeout=0.1)
                
#                 # Skip old actions
#                 if time.time() - timestamp > 0.2:
#                     continue
                    
#                 self._execute_immediate(action_type, cursor)
#                 time.sleep(0.01)  # Small delay
                
#             except Empty:
#                 # Process accumulated scrolling
#                 self._process_scroll_batch(None)
#                 continue
    
#     def _execute_immediate(self, action_type, cursor):
#         """Execute action immediately"""
#         if action_type == 1:
#             cursor.click()
#         elif action_type in [2, 3]:
#             scroll_delta = -100 if action_type == 2 else 100
#             cursor.scroll(scroll_delta)
    
#     def _process_scroll_batch(self, cursor):
#         """Process accumulated scroll actions"""
#         current_time = time.time()
#         if (self.scroll_accumulator != 0 and 
#             current_time - self.last_scroll_time > self.scroll_batch_time):
            
#             if cursor:
#                 cursor.scroll(self.scroll_accumulator)
#             self.scroll_accumulator = 0
#             self.last_scroll_time = current_time
    
#     def execute_action(self, output, cursor):
#         """Main method to call from your code"""
#         current_time = time.time()
#         action_type = output[0]
        
#         # Rate limiting
#         if current_time - self.last_action_time < self.min_interval:
#             return
        
#         if self.use_threading:
#             # Add to queue for background processing
#             try:
#                 self.action_queue.put_nowait((action_type, cursor, current_time))
#             except:
#                 pass  # Queue full, skip this action
#         else:
#             # Synchronous processing with batching
#             if action_type == 1:
#                 cursor.click()
#             elif action_type in [2, 3]:
#                 scroll_delta = -100 if action_type == 2 else 100
#                 self.scroll_accumulator += scroll_delta
#                 self._process_scroll_batch(cursor)
        
#         self.last_action_time = current_time



class OptimizedMouseController:
    def __init__(self, min_interval=0.01, scroll_batch_time=0.05):
        self.min_interval = min_interval
        self.scroll_batch_time = scroll_batch_time
        self.last_action_time = 0
        
        # Optimized scroll batching
        self.scroll_accumulator = 0
        self.last_scroll_time = 0
        self.pending_scroll = False
        
        # Optimized threading with lock-free approach
        self.action_queue = deque(maxlen=50)  # Lock-free deque with size limit
        self.queue_lock = threading.Lock()
        self.running = True
        
        # Use daemon thread with optimized worker
        self.worker_thread = threading.Thread(target=self._optimized_worker, daemon=True)
        self.worker_thread.start()
        
        # Position tracking for movement optimization
        self.last_pos = (0, 0)
        self.position_threshold = 5  # Minimum pixel movement
        
        # Windows API optimization (if on Windows)
        self.use_win_api = self._init_win_api()
    
    def _init_win_api(self):
        """Initialize Windows API for faster mouse operations"""
        try:
            self.user32 = ctypes.windll.user32
            self.SetCursorPos = self.user32.SetCursorPos
            self.GetCursorPos = self.user32.GetCursorPos
            self.mouse_event = self.user32.mouse_event
            return True
        except:
            return False
    
    def _optimized_worker(self):
        """Highly optimized background worker"""
        local_scroll = 0
        last_process_time = time.perf_counter()
        
        while self.running:
            current_time = time.perf_counter()
            processed_any = False
            
            # Process multiple actions per iteration
            for _ in range(10):  # Process up to 10 actions per loop
                try:
                    with self.queue_lock:
                        if self.action_queue:
                            action_type, data, timestamp = self.action_queue.popleft()
                        else:
                            break
                    
                    # Skip stale actions (older than 100ms)
                    if current_time - timestamp > 0.1:
                        continue
                    
                    if action_type == 0:  # Move
                        self._fast_move(data)
                    elif action_type == 1:  # Click
                        self._fast_click(data)
                    elif action_type in [2, 3]:  # Scroll
                        scroll_delta = -20 if action_type == 2 else 20
                        local_scroll += scroll_delta
                    
                    processed_any = True
                    
                except Exception:
                    continue
            
            # Batch process scrolling
            if local_scroll != 0 and current_time - last_process_time > self.scroll_batch_time:
                self._fast_scroll(local_scroll)
                local_scroll = 0
                last_process_time = current_time
            
            # Adaptive sleep based on queue size
            if not processed_any:
                time.sleep(0.001)  # 1ms when idle
            elif len(self.action_queue) > 20:
                continue  # No sleep when busy
            else:
                time.sleep(0.0001)  # 0.1ms when moderately busy
    
    def _fast_move(self, pos):
        """Optimized mouse movement"""
        pos_x, pos_y = pos
        
        # Skip small movements to reduce jitter
        dx = abs(pos_x - self.last_pos[0])
        dy = abs(pos_y - self.last_pos[1])
        # if dx < self.position_threshold and dy < self.position_threshold:
        #     return
        
        if self.use_win_api:
            self.SetCursorPos(int(pos_x), int(pos_y))
        else:
            # Fallback to original method
            import pyautogui
            pyautogui.moveTo(int(pos_x), int(pos_y), duration=0)
        
        self.last_pos = (pos_x, pos_y)
    
    def _fast_click(self, cursor):
        """Optimized clicking"""
        if self.use_win_api:
            # Direct Windows API calls for faster clicking
            self.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
            self.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
        else:
            cursor.click()
    
    def _fast_scroll(self, delta):
        """Optimized scrolling"""
        if self.use_win_api:
            self.mouse_event(0x0800, 0, 0, delta, 0)  # MOUSEEVENTF_WHEEL
        else:
            import pyautogui
            pyautogui.scroll(delta // 120)  # Convert to scroll units
    
    def move_to(self, pos_x, pos_y):
        """Optimized move operation"""
        current_time = time.perf_counter()
        
        # Rate limiting with higher precision timer
        # if current_time - self.last_action_time < self.min_interval:
        #     return
        
        # Direct execution for movement (no queuing for responsiveness)
        self._fast_move((pos_x, pos_y))
        self.last_action_time = current_time
    
    def execute_action(self, output, cursor=None):
        """Main execution method with optimizations"""
        current_time = time.perf_counter()
        action_type = output[0]
        
        # Rate limiting
        if current_time - self.last_action_time < self.min_interval:
            return
        
        # Queue management with overflow protection
        if len(self.action_queue) >= 45:  # Leave some headroom
            # Remove oldest actions to make space
            with self.queue_lock:
                for _ in range(5):
                    if self.action_queue:
                        self.action_queue.popleft()
        
        # Add action to queue
        try:
            data = None
            if action_type == 0 and len(output) >= 3:  # Move
                data = (output[1], output[2])
            elif action_type == 1:  # Click
                data = cursor
            
            with self.queue_lock:
                self.action_queue.append((action_type, data, current_time))
                
        except Exception:
            pass  # Ignore queue errors
        
        self.last_action_time = current_time
    
    def cleanup(self):
        """Clean shutdown"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.1)
    
    def __del__(self):
        """Destructor"""
        self.cleanup()


# Usage example:
controller = OptimizedMouseController(min_interval=0.005, scroll_batch_time=0.03)
# 
# # For mouse movement (replaces cursor.moveTo)
# controller.move_to(pos_x, pos_y)
# 
# # For other actions
# controller.execute_action([1], cursor)  # Click
# controller.execute_action([2], cursor)  # Scroll up
# controller.execute_action([3], cursor)  # Scroll down
# Usage
#controller = OptimizedMouseController(min_interval=0.03)  # Max 33 actions per second
# Example usage
detector = HandDetector()
class TFLiteModel:
    def __init__(self, model_path):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict(self, input_data):
        # Prepare input data
        input_data = np.array(input_data, dtype=np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output
model = TFLiteModel('hand_gesture_model_lite.tflite')
print("Hand Detection Application")
print("=" * 30)
print("Controls:")
print("- Press 'q' to quit")
print("- Press 's' to save current hand positions")
print("- Press 'p' to print hand positions to console")
print("=" * 30)
pos_x, pos_y = 0, 0
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Flip frame horiqzontally for mirror effect
    #frame = cv2.flip(frame, 1)
    
    # Run hand detection
    hand_data = detector.run_webcam(frame)
    temp1 = preprocess_hands_data([hand_data])
    temp1 = [hand['points'] for hand in temp1]
    temp1 = np.array(temp1)
    if temp1.shape[0]!=0:
        position = apply_warp_to_points(temp1[0,8:9,:2],transform_matrix)
        mines = np.min(temp1, axis=1)
        maxes = np.max(temp1, axis=1)
        temp1 = (temp1 - mines[:, np.newaxis]) / (maxes - mines)[:, np.newaxis]
        
        output = model.predict(temp1)
        output = np.argmax(output, axis=1)
        controller.execute_action(output, cursor)
        pos_x, pos_y = 0.5*position[0, 0]*1920+0.5*pos_x, 0.5*position[0, 1]*1080+0.5*pos_y
        controller.move_to(int(pos_x), int(pos_y))
    # Display the processed frame
    #cv2.imshow('Hand Detection', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        if hand_data:
            detector.print_hand_positions(hand_data)
        else:
            print("No hands detected!")
cap.release()
cv2.destroyAllWindows()