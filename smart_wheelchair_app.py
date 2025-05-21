import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import PIL.Image
import threading
import time
import heapq
import re
import speech_recognition as sr

# PyQt5 Imports
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QMessageBox, QFileDialog, QLineEdit,
    QInputDialog, QStatusBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt

# Suppress matplotlib user warnings about tight_layout etc.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Global Model Loading ---
# Determine the device to use (CUDA for NVIDIA GPU, MPS for Apple Silicon, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon (MPS) GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# Load YOLOv5 model globally once
yolo_model = None
try:
    print("Loading YOLOv5 model (yolov5s)...")
    # Force reload=True to ensure it's not using a stale cached version if issues arise
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
    yolo_model.to(device)
    yolo_model.eval()
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    print("YOLOv5 functionality will be unavailable.")

# Load Faster R-CNN models as needed
fasterrcnn_static_model = None
fasterrcnn_video_model = None
try:
    print("Loading Faster R-CNN models...")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.transforms import functional as F
    fasterrcnn_static_model = fasterrcnn_resnet50_fpn(pretrained=True)
    fasterrcnn_static_model.to(device)
    fasterrcnn_static_model.eval()
    print("Faster R-CNN ResNet50 model loaded.")

    fasterrcnn_video_model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    fasterrcnn_video_model.to(device)
    fasterrcnn_video_model.eval()
    print("Faster R-CNN MobileNetV3 model loaded.")
except ImportError as e:
    print(f"Error loading Faster R-CNN models: {e}. Ensure torchvision is installed.")
    print("Faster R-CNN functionality will be unavailable.")
except Exception as e:
    print(f"An unexpected error occurred while loading Faster R-CNN models: {e}")
    print("Faster R-CNN functionality will be unavailable.")

# --- Helper function to get class names for Faster R-CNN (COCO dataset) ---
# Faster R-CNN models are typically trained on COCO dataset.
# This list maps numeric labels to human-readable class names.
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def get_coco_class_name(label_id):
    try:
        return COCO_INSTANCE_CATEGORY_NAMES[label_id]
    except IndexError:
        return f"Unknown ID:{label_id}"

# --- Camera Worker Thread for Obstacle Detection ---
class CameraWorker(QThread):
    # Signals to send data back to the GUI thread
    image_update = pyqtSignal(QImage)
    movement_command = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, yolo_model):
        super().__init__()
        self._run_flag = True
        self.yolo_model = yolo_model
        self.cap = None

    def run(self):
        if self.yolo_model is None:
            self.status_update.emit("Error: YOLOv5 model not loaded. Please restart with model available.")
            self._run_flag = False
            return

        self.cap = cv2.VideoCapture(0) # 0 for default camera
        if not self.cap.isOpened():
            self.status_update.emit("Error: Could not open camera. Check if another app is using it or if drivers are installed.")
            self._run_flag = False
            return

        self.status_update.emit("Camera feed started. Detecting obstacles...")

        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                # If frame not read, try to re-open camera or break
                print("Failed to read frame, attempting to re-open camera...")
                self.cap.release()
                time.sleep(1) # Wait a bit before retrying
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.status_update.emit("Error: Camera lost and could not be re-opened.")
                    self._run_flag = False
                continue

            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = None
            try:
                with torch.no_grad():
                    # YOLOv5 expects RGB images
                    results = self.yolo_model(frame_rgb)
            except Exception as e:
                self.status_update.emit(f"Error during YOLOv5 inference: {e}")
                continue


            detections = results.xyxy[0].cpu().numpy()
            # Define obstacle classes for the wheelchair
            # These are common objects YOLOv5 can detect that might be obstacles
            obstacle_classes = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'train', 'truck',
                                'chair', 'couch', 'potted plant', 'dining table', 'bed', 'toilet',
                                'tv', 'laptop', 'keyboard', 'cell phone', 'microwave', 'oven', 'sink',
                                'refrigerator', 'book', 'vase', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl']
            obstacles = []

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                class_name = self.yolo_model.names[int(cls)]
                if class_name in obstacle_classes and conf > 0.5: # Confidence threshold for display
                    obstacles.append([x1, y1, x2, y2])
                    # Draw bounding box on the original frame (BGR format for OpenCV drawing)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            movement = self.decide_movement(obstacles, w)
            self.movement_command.emit(movement)

            # Convert frame (which is BGR from OpenCV) to QImage for display
            # QImage.Format_BGR888 is correct for OpenCV BGR frames
            qt_image = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
            self.image_update.emit(qt_image)

            time.sleep(0.03) # Approximate 30 FPS

        self.cap.release()
        self.status_update.emit("Camera feed stopped.")


    def decide_movement(self, detections, frame_width):
        # Simple logic: if obstacles are in the center, stop.
        # Otherwise, if more obstacles on left, move right, and vice-versa.
        # If no obstacles, move forward.
        left_count = 0
        right_count = 0
        center_count = 0

        for det in detections:
            x_center = (det[0] + det[2]) / 2 # Center X coordinate of the bounding box
            if x_center < frame_width / 3: # Left third of the frame
                left_count += 1
            elif x_center > 2 * frame_width / 3: # Right third of the frame
                right_count += 1
            else: # Middle third of the frame
                center_count += 1

        if center_count > 0:
            return "stop"
        elif left_count > right_count:
            return "move_right"
        elif right_count > left_count:
            return "move_left"
        else:
            return "move_forward"

    def stop(self):
        self._run_flag = False
        # Wait for the thread to finish its current loop iteration and exit cleanly
        self.wait()

# --- Main Application Class ---
class WheelchairSimulationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Wheelchair Simulation System")
        self.setGeometry(100, 100, 1200, 800) # Initial window size and position

        # Main widget and layout for the central area of the QMainWindow
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Tab Widget (Notebook equivalent in Tkinter)
        self.notebook = QTabWidget(self)
        self.main_layout.addWidget(self.notebook)

        # Create individual tabs as QWidget instances
        self.tab_obstacle_detection = QWidget()
        self.tab_manual_navigation = QWidget()
        self.tab_path_planning = QWidget()
        self.tab_static_detection = QWidget()
        self.tab_video_detection = QWidget()
        self.tab_voice_control = QWidget()

        # Add tabs to the QTabWidget
        self.notebook.addTab(self.tab_obstacle_detection, "Obstacle Detection")
        self.notebook.addTab(self.tab_manual_navigation, "Manual Navigation")
        self.notebook.addTab(self.tab_path_planning, "Path Planning")
        self.notebook.addTab(self.tab_static_detection, "Static Image Detection")
        self.notebook.addTab(self.tab_video_detection, "Video Detection")
        self.notebook.addTab(self.tab_voice_control, "Voice Control")

        # Status Bar at the bottom of the main window
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Initialize worker threads to None
        self.camera_worker_thread = None
        self.video_processing_thread = None
        self.voice_control_thread = None

        # Path planning specific attributes
        self.floor_plan = None # The original loaded map for path planning
        self.buffered_floor_plan = None # The map with obstacles inflated for path planning
        self.obstacle_buffer_size = 2 # Default buffer size in pixels
        
        # Path animation attributes
        self.path_animation_timer = QTimer(self)
        self.path_animation_timer.timeout.connect(self.animate_path_step)
        self.current_animated_path = None
        self.current_path_index = 0
        self.animated_dot = None # The moving dot on the path planning map

        # Voice control map attributes
        self.voice_map = None
        self.voice_map_position = None
        self.voice_map_fig = None
        self.voice_map_ax = None
        self.voice_map_canvas = None
        self.voice_dot = None
        self.voice_move_timer = QTimer(self) # Timer for continuous movement on voice map
        self.voice_move_timer.timeout.connect(self.animate_voice_move_step)
        self.voice_current_direction = None # Stores the current direction for continuous movement


        # Setup content for each tab
        self.setup_obstacle_detection_tab()
        self.setup_manual_navigation_tab()
        self.setup_path_planning_tab()
        self.setup_static_detection_tab()
        self.setup_video_detection_tab()
        self.setup_voice_control_tab()

    # ================== Obstacle Detection tab ==================
    def setup_obstacle_detection_tab(self):
        layout = QVBoxLayout(self.tab_obstacle_detection)

        # Control frame for buttons and movement command label
        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)

        self.start_obstacle_btn = QPushButton("Start Camera")
        self.start_obstacle_btn.clicked.connect(self.start_obstacle_detection)
        control_layout.addWidget(self.start_obstacle_btn)

        self.stop_obstacle_btn = QPushButton("Stop Camera")
        self.stop_obstacle_btn.clicked.connect(self.stop_obstacle_detection)
        self.stop_obstacle_btn.setEnabled(False) # Initially disabled
        control_layout.addWidget(self.stop_obstacle_btn)

        self.movement_label = QLabel("Movement Command: None")
        self.movement_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        control_layout.addWidget(self.movement_label)

        control_layout.addStretch(1) # Pushes widgets to the left

        layout.addWidget(control_frame)

        # Label to display the camera feed
        self.obstacle_video_label = QLabel("Camera Feed")
        self.obstacle_video_label.setAlignment(Qt.AlignCenter)
        self.obstacle_video_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        layout.addWidget(self.obstacle_video_label, 1) # Expands vertically

        info_text = ("This tab uses YOLOv5 for real-time obstacle detection from your camera.\n"
                     "Press 'Start Camera' to begin detection. Press 'Stop Camera' to stop.")
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True) # Allow text to wrap
        layout.addWidget(info_label)

    def start_obstacle_detection(self):
        if self.camera_worker_thread and self.camera_worker_thread.isRunning():
            return # Already running

        if yolo_model is None:
            self.status_bar.showMessage("Error: YOLOv5 model not loaded.")
            QMessageBox.critical(self, "Model Error", "YOLOv5 model could not be loaded. Check console for details.")
            return

        # Create and start the camera worker thread
        self.camera_worker_thread = CameraWorker(yolo_model)
        # Connect signals from the worker to slots in the main GUI thread
        self.camera_worker_thread.image_update.connect(self.update_obstacle_image)
        self.camera_worker_thread.movement_command.connect(self.update_movement_command)
        self.camera_worker_thread.status_update.connect(self.status_bar.showMessage)

        self.start_obstacle_btn.setEnabled(False)
        self.stop_obstacle_btn.setEnabled(True)
        self.camera_worker_thread.start()

    def stop_obstacle_detection(self):
        if self.camera_worker_thread and self.camera_worker_thread.isRunning():
            self.camera_worker_thread.stop() # Request the thread to stop
            self.camera_worker_thread.wait() # Wait for the thread to actually finish
            self.start_obstacle_btn.setEnabled(True)
            self.stop_obstacle_btn.setEnabled(False)
            self.obstacle_video_label.clear() # Clear video display
            self.movement_label.setText("Movement Command: None")
            self.status_bar.showMessage("Stopping camera...")


    @QtCore.pyqtSlot(QImage) # Decorator for clarity, though not strictly required
    def update_obstacle_image(self, image):
        # Scale QImage to fit the QLabel, maintaining aspect ratio
        label_size = self.obstacle_video_label.size()
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.obstacle_video_label.setPixmap(scaled_pixmap)

    @QtCore.pyqtSlot(str)
    def update_movement_command(self, command):
        self.movement_label.setText(f"Movement Command: {command.replace('_', ' ').capitalize()}")

    # ================== Manual Navigation tab ==================
    def setup_manual_navigation_tab(self):
        layout = QVBoxLayout(self.tab_manual_navigation)

        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)

        self.load_map_btn = QPushButton("Load Floor Map")
        self.load_map_btn.clicked.connect(self.load_floor_map)
        control_layout.addWidget(self.load_map_btn)
        control_layout.addStretch(1) # Pushes button to left

        layout.addWidget(control_frame)

        # Frame for the Matplotlib map
        self.map_frame = QWidget()
        map_layout = QVBoxLayout(self.map_frame)
        layout.addWidget(self.map_frame, 1) # Expand vertically

        self.map_fig = Figure(figsize=(6, 6), dpi=100)
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_canvas = FigureCanvas(self.map_fig)
        map_layout.addWidget(self.map_canvas)

        # Movement control buttons using a QGridLayout
        movement_grid_widget = QWidget()
        movement_grid_layout = QtWidgets.QGridLayout(movement_grid_widget)
        layout.addWidget(movement_grid_widget)

        # Up button
        self.up_btn = QPushButton("↑")
        self.up_btn.setFixedSize(50, 50) # Make buttons square for consistent look
        self.up_btn.clicked.connect(lambda: self.move_manual('up'))
        movement_grid_layout.addWidget(self.up_btn, 0, 1)

        # Left button
        self.left_btn = QPushButton("←")
        self.left_btn.setFixedSize(50, 50)
        self.left_btn.clicked.connect(lambda: self.move_manual('left'))
        movement_grid_layout.addWidget(self.left_btn, 1, 0)

        # Down button
        self.down_btn = QPushButton("↓")
        self.down_btn.setFixedSize(50, 50)
        self.down_btn.clicked.connect(lambda: self.move_manual('down'))
        movement_grid_layout.addWidget(self.down_btn, 1, 1)

        # Right button
        self.right_btn = QPushButton("→")
        self.right_btn.setFixedSize(50, 50)
        self.right_btn.clicked.connect(lambda: self.move_manual('right'))
        movement_grid_layout.addWidget(self.right_btn, 1, 2)

        info_text = "This tab allows manual navigation on a floor map.\nLoad a floor map image, then use the directional buttons to move."
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.manual_position = None
        self.floor_map = None
        self.dot = None # Matplotlib plot object for the current position

    def load_floor_map(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Floor Map Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if not file_path:
            return
        try:
            im = PIL.Image.open(file_path).convert("L") # Convert to grayscale
            self.floor_map = np.array(im)
            # Convert to binary map: obstacles=1 (dark), free=0 (light)
            self.floor_map = np.where(self.floor_map < 128, 1, 0)
            self.map_ax.clear()
            self.map_ax.imshow(self.floor_map, cmap="gray")

            # Set initial position to center, then find closest valid free space
            rows, cols = self.floor_map.shape
            initial_r, initial_c = rows // 2, cols // 2
            self.manual_position = (initial_r, initial_c)

            if self.floor_map[self.manual_position] != 0: # If center is an obstacle
                found = False
                # Search outwards in increasing radii for a free spot
                for radius in range(max(rows, cols)):
                    for dr in range(-radius, radius+1):
                        for dc in range(-radius, radius+1):
                            r = initial_r + dr
                            c = initial_c + dc
                            if 0 <= r < rows and 0 <= c < cols: # Check bounds
                                if self.floor_map[r, c] == 0: # Found a free spot
                                    self.manual_position = (r, c)
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    QMessageBox.warning(self, "Map Error", "Could not find a free starting position on the map.")
                    self.status_bar.showMessage("Error: No free starting position found on map.")
                    return

            # Plot the current position dot
            self.dot, = self.map_ax.plot(self.manual_position[1], self.manual_position[0], 'ro', markersize=8)
            self.map_canvas.draw() # Redraw the canvas
            self.status_bar.showMessage(f"Loaded floor map: {file_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading floor map: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading floor map: {str(e)}")

    def move_manual(self, direction):
        if self.floor_map is None or self.manual_position is None:
            self.status_bar.showMessage("Please load a floor map first.")
            return

        r, c = self.manual_position
        new_pos = (r, c) # Initialize new position

        # Calculate new position based on direction
        if direction == 'up':
            new_pos = (r-1, c)
        elif direction == 'down':
            new_pos = (r+1, c)
        elif direction == 'left':
            new_pos = (r, c-1)
        elif direction == 'right':
            new_pos = (r, c+1)
        else:
            return # Invalid direction

        # Check if the new position is valid (within bounds and not an obstacle)
        if self.is_valid_position_on_map(new_pos, self.floor_map):
            self.manual_position = new_pos
            self.dot.set_data([new_pos[1]], [new_pos[0]]) # Update dot's coordinates
            self.map_canvas.draw() # Redraw map to show new dot position
            self.status_bar.showMessage(f"Moved {direction}")
        else:
            self.status_bar.showMessage("Move blocked by obstacle or wall.")

    def is_valid_position_on_map(self, pos, current_map):
        r, c = pos
        rows, cols = current_map.shape
        # Check bounds and if the cell is free (0)
        return 0 <= r < rows and 0 <= c < cols and current_map[r, c] == 0

    # ================== Path Planning tab ==================
    def setup_path_planning_tab(self):
        layout = QVBoxLayout(self.tab_path_planning)

        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)

        self.load_path_map_btn = QPushButton("Load Floor Map")
        self.load_path_map_btn.clicked.connect(self.load_path_map)
        control_layout.addWidget(self.load_path_map_btn)

        self.mark_point_btn = QPushButton("Mark Point")
        self.mark_point_btn.clicked.connect(self.toggle_marking)
        control_layout.addWidget(self.mark_point_btn)

        self.name_points_btn = QPushButton("Name Points")
        self.name_points_btn.clicked.connect(self.name_points)
        control_layout.addWidget(self.name_points_btn)

        self.find_path_btn = QPushButton("Find Path")
        self.find_path_btn.clicked.connect(self.find_path)
        control_layout.addWidget(self.find_path_btn)

        self.clear_points_btn = QPushButton("Clear Points")
        self.clear_points_btn.clicked.connect(self.clear_points)
        control_layout.addWidget(self.clear_points_btn)

        # Buffer size input
        control_layout.addWidget(QLabel("Buffer Size:"))
        self.buffer_size_input = QLineEdit(str(self.obstacle_buffer_size))
        self.buffer_size_input.setFixedWidth(50)
        self.buffer_size_input.setValidator(QtGui.QIntValidator(0, 10)) # Limit buffer to 0-10 pixels
        control_layout.addWidget(self.buffer_size_input)

        self.apply_buffer_btn = QPushButton("Apply Buffer")
        self.apply_buffer_btn.clicked.connect(self.apply_obstacle_buffer)
        control_layout.addWidget(self.apply_buffer_btn)


        control_layout.addStretch(1) # Pushes buttons to the left
        layout.addWidget(control_frame)

        # Frame for the Matplotlib path planning map
        self.path_map_frame = QWidget()
        path_map_layout = QVBoxLayout(self.path_map_frame)
        layout.addWidget(self.path_map_frame, 1)

        self.path_fig = Figure(figsize=(6,6), dpi=100)
        self.path_ax = self.path_fig.add_subplot(111)
        self.path_canvas = FigureCanvas(self.path_fig)
        path_map_layout.addWidget(self.path_canvas)

        info_text = ("Plan paths on a floor map. Load a floor map, mark points by clicking,\n"
                     "name them and compute path between points using A* algorithm.\n"
                     "Adjust 'Buffer Size' to keep paths away from walls/obstacles.")
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.point_marking_enabled = False
        self.marked_points = {}   # Dictionary: name -> (row, col)
        self.marked_positions = []   # List of (row, col) for plotting circles
        self.point_circles = [] # Matplotlib plot objects for circles
        self.text_labels = []   # Matplotlib text objects for names
        self.current_path_line = None # Matplotlib plot object for the path line

        # Connect mouse click event on the map canvas
        self.path_canvas.mpl_connect('button_press_event', self.on_map_click)

    def load_path_map(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Floor Map Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if not file_path:
            return
        try:
            im = PIL.Image.open(file_path).convert("L")
            self.floor_plan = np.array(im)
            self.floor_plan = np.where(self.floor_plan < 128, 1, 0)   # 1: obstacle, 0: free

            # Apply initial buffer when map is loaded
            self.apply_obstacle_buffer() # This will also clear and redraw

            self.status_bar.showMessage(f"Loaded floor map for path planning: {file_path}")

        except Exception as e:
            self.status_bar.showMessage(f"Error loading floor map: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading floor map: {str(e)}")

    def apply_obstacle_buffer(self):
        if self.floor_plan is None:
            self.status_bar.showMessage("Load a floor map first to apply buffer.")
            return

        try:
            new_buffer_size = int(self.buffer_size_input.text())
            if new_buffer_size < 0:
                raise ValueError("Buffer size cannot be negative.")
            self.obstacle_buffer_size = new_buffer_size
        except ValueError:
            self.status_bar.showMessage("Invalid buffer size. Please enter an integer.")
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for buffer size (e.g., 0, 1, 2).")
            return

        # Create a copy to avoid modifying the original floor_plan directly
        self.buffered_floor_plan = np.copy(self.floor_plan)

        if self.obstacle_buffer_size > 0:
            # Create a kernel for dilation (e.g., a square kernel)
            # The size of the kernel determines the "reach" of the buffer
            kernel = np.ones((self.obstacle_buffer_size * 2 + 1, self.obstacle_buffer_size * 2 + 1), np.uint8)
            
            # Dilate the obstacles (where value is 1)
            # Invert the map for dilation: obstacles become 255, free space 0
            temp_map = (self.floor_plan * 255).astype(np.uint8)
            dilated_map = cv2.dilate(temp_map, kernel, iterations=1)
            
            # Convert back to binary (1 for obstacle, 0 for free)
            self.buffered_floor_plan = np.where(dilated_map > 0, 1, 0)

        # Clear existing points and paths and redraw the map with the new buffer
        self.clear_points() # This also redraws the map
        self.status_bar.showMessage(f"Obstacle buffer applied: {self.obstacle_buffer_size} pixels.")


    def toggle_marking(self):
        self.point_marking_enabled = not self.point_marking_enabled
        status_msg = f"Point marking {'enabled' if self.point_marking_enabled else 'disabled'}. Click on map to mark points."
        self.status_bar.showMessage(status_msg)

    def on_map_click(self, event):
        if not self.point_marking_enabled:
            return
        if self.floor_plan is None: # Check original floor_plan for marking points
            self.status_bar.showMessage("Please load a floor map first.")
            return
        if event.xdata is None or event.ydata is None: # Click outside axes
            return

        r = int(event.ydata) # Row (Y-coordinate)
        c = int(event.xdata) # Column (X-coordinate)

        # Check if click is within map bounds on the ORIGINAL map
        if 0 <= r < self.floor_plan.shape[0] and 0 <= c < self.floor_plan.shape[1]:
            if self.floor_plan[r,c] == 0: # Check if the clicked point is a free space on original map
                self.marked_positions.append((r,c))
                circle = self.path_ax.plot(c, r, 'ro', markersize=8)[0] # Plot red circle
                self.point_circles.append(circle)
                self.path_canvas.draw()
                self.status_bar.showMessage(f"Marked point at ({r}, {c}). Now name it.")
            else:
                self.status_bar.showMessage("Cannot mark point on obstacle/wall (original map).")
        else:
            self.status_bar.showMessage("Click out of map bounds.")


    def name_points(self):
        if not self.marked_positions:
            self.status_bar.showMessage("No points to name. Mark points first.")
            return

        # Clear existing named points to re-populate
        self.marked_points.clear()
        # Clear existing text labels from the map
        for txt in self.text_labels:
            txt.remove()
        self.text_labels.clear()

        for idx, pos in enumerate(self.marked_positions):
            # QInputDialog for getting point name from user
            name, ok = QInputDialog.getText(self, "Name Point", f"Enter name for point {idx+1} (position {pos}):")
            if ok and name:
                self.marked_points[name] = pos
            else:
                # Assign a default name if user cancels or enters empty string
                self.marked_points[f"Point {idx+1}"] = pos

        # Display point names on map
        for name, pos in self.marked_points.items():
            r,c = pos
            txt = self.path_ax.text(c, r, name, color="blue", fontsize=10, ha='center', va='bottom')
            self.text_labels.append(txt)
        self.path_canvas.draw()
        self.status_bar.showMessage(f"Named {len(self.marked_points)} points.")

    def clear_points(self):
        self.marked_points.clear()
        self.marked_positions.clear()
        # Remove all plotted circles
        for circle in self.point_circles:
            circle.remove()
        self.point_circles.clear()
        # Remove all text labels
        for txt in self.text_labels:
            txt.remove()
        self.text_labels.clear()
        # Remove the path line if it exists
        if self.current_path_line:
            self.current_path_line.remove()
            self.current_path_line = None
        
        # Stop and clear animation related elements
        self.path_animation_timer.stop()
        self.current_animated_path = None
        self.current_path_index = 0
        if self.animated_dot:
            self.animated_dot.remove()
            self.animated_dot = None
            # Remove legend if it was added
            if self.path_ax.legend_ is not None:
                self.path_ax.legend().remove()
                self.path_ax.legend([]) # Clear legend handles

        # Redraw the map to clear all markings
        self.path_ax.clear()
        # Display the buffered map if available, otherwise the original floor plan
        if self.buffered_floor_plan is not None:
            self.path_ax.imshow(self.buffered_floor_plan, cmap="gray")
        elif self.floor_plan is not None:
            self.path_ax.imshow(self.floor_plan, cmap="gray")
        self.path_canvas.draw()
        self.status_bar.showMessage("Cleared all points and paths.")

    def find_path(self):
        if not self.marked_points or len(self.marked_points) < 2:
            self.status_bar.showMessage("Need at least two named points to find path.")
            QMessageBox.warning(self, "Insufficient Points", "Please mark and name at least two points on the map to find a path.")
            return
        
        if self.buffered_floor_plan is None:
            self.status_bar.showMessage("Please load a floor map and apply buffer first.")
            QMessageBox.warning(self, "Map Not Ready", "Please load a floor map and ensure the obstacle buffer is applied.")
            return

        names = list(self.marked_points.keys())
        # Use QInputDialog to let user select start and end points
        start_name, ok_start = QInputDialog.getItem(self, "Select Start Point", "Choose start point:", names, 0, False)
        if not ok_start or not start_name: return # User cancelled

        end_name, ok_end = QInputDialog.getItem(self, "Select End Point", "Choose end point:", names, 0, False)
        if not ok_end or not end_name: return # User cancelled

        if start_name not in self.marked_points or end_name not in self.marked_points:
            self.status_bar.showMessage("Invalid point names selected.")
            QMessageBox.warning(self, "Invalid Selection", "Selected start or end point name is not recognized.")
            return

        start = self.marked_points[start_name]
        end = self.marked_points[end_name]

        # Check if start or end points are within the buffered obstacles
        if self.buffered_floor_plan[start[0], start[1]] == 1:
            QMessageBox.warning(self, "Invalid Start Point", "Start point is within a buffered obstacle. Please choose a free space.")
            self.status_bar.showMessage("Start point is in buffered obstacle.")
            return
        if self.buffered_floor_plan[end[0], end[1]] == 1:
            QMessageBox.warning(self, "Invalid End Point", "End point is within a buffered obstacle. Please choose a free space.")
            self.status_bar.showMessage("End point is in buffered obstacle.")
            return


        self.status_bar.showMessage(f"Finding path from {start_name} to {end_name}...")
        path = self.astar_path(start, end) # Call A* algorithm

        if path is None:
            self.status_bar.showMessage("No path found between points.")
            QMessageBox.warning(self, "Path Not Found", "No valid path could be found between the selected points. Check for obstacles or increase buffer size.")
            return

        # Clear previous path line and animated dot
        if self.current_path_line:
            self.current_path_line.remove()
            self.current_path_line = None
        if self.animated_dot:
            self.animated_dot.remove()
            self.animated_dot = None
            if self.path_ax.legend_ is not None:
                self.path_ax.legend().remove()
                self.path_ax.legend([])

        # Redraw map, points, and names
        self.path_ax.clear()
        # Always display the buffered map for path planning visualization
        self.path_ax.imshow(self.buffered_floor_plan, cmap="gray")
        for pos in self.marked_positions:
            self.path_ax.plot(pos[1], pos[0], 'ro', markersize=8)
        for name, pos in self.marked_points.items():
            self.path_ax.text(pos[1], pos[0], name, color="blue", fontsize=10, ha='center', va='bottom')

        # Draw the new path
        path_r = [p[0] for p in path]
        path_c = [p[1] for p in path]
        self.current_path_line, = self.path_ax.plot(path_c, path_r, 'g-', linewidth=2, marker='o', markersize=4, markerfacecolor='green')
        
        # Start the animation
        self.current_animated_path = path
        self.current_path_index = 0
        self.path_animation_timer.start(100) # Animate every 100 ms (adjust speed as needed)

        self.status_bar.showMessage(f"Path found from {start_name} to {end_name} (length {len(path)} cells). Starting animation...")

    def animate_path_step(self):
        """
        Animates the wheelchair moving along the calculated path.
        """
        if self.current_animated_path is None or self.current_path_index >= len(self.current_animated_path):
            self.path_animation_timer.stop()
            self.status_bar.showMessage("Path animation finished.")
            return

        r, c = self.current_animated_path[self.current_path_index]

        if self.animated_dot is None:
            # Create the animated dot if it doesn't exist
            self.animated_dot, = self.path_ax.plot(c, r, 'yo', markersize=10, label='Wheelchair') # Yellow dot
            self.path_ax.legend() # Show legend for the dot
        else:
            # Update its position
            self.animated_dot.set_data([c], [r])
        
        self.path_canvas.draw_idle() # Redraw only the changed parts for efficiency
        self.status_bar.showMessage(f"Moving to ({r}, {c})...")
        self.current_path_index += 1

    def astar_path(self, start, goal):
        """
        A* pathfinding algorithm on the buffered floor plan.
        Returns a list of (row, col) tuples representing the path, or None if no path.
        """
        def heuristic(a, b):
            # Manhattan distance heuristic (L1 norm)
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        rows, cols = self.buffered_floor_plan.shape # Use buffered map dimensions
        
        # Priority queue: (f_cost, g_cost, current_node, parent_node_for_reconstruction)
        # f_cost = g_cost + h_cost
        open_set = []
        # Initial push: (f_cost, g_cost, current_node, parent_node)
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None)) 
        
        came_from = {start: None} # To reconstruct path: child_node -> parent_node
        cost_so_far = {start: 0} # g_cost: cost from start to current_node

        # Define possible movements (8 directions: horizontal, vertical, and diagonal)
        # (dr, dc) for (row_change, col_change)
        # Cost for horizontal/vertical move = 1
        # Cost for diagonal move = sqrt(2) approx 1.414
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # 4-directional (horizontal/vertical)
            (1, 1), (1, -1), (-1, 1), (-1, -1) # 8-directional (diagonals)
        ]
        
        # Costs for each direction
        direction_costs = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
        
        turn_penalty = 0.5 # A small penalty for changing direction, adjust as needed

        while open_set:
            # Pop node with the lowest f_cost
            f_cost, g_cost, current_node, parent_of_current = heapq.heappop(open_set)

            # If we reached the goal, reconstruct and return the path
            if current_node == goal:
                path = []
                temp_node = current_node
                while temp_node is not None: # Reconstruct path from goal back to start
                    path.append(temp_node)
                    temp_node = came_from[temp_node]
                path.reverse() # Path is built backwards, so reverse it
                return path

            # Explore neighbors
            for i, (dr, dc) in enumerate(directions):
                next_r, next_c = current_node[0] + dr, current_node[1] + dc
                next_node = (next_r, next_c)
                move_cost = direction_costs[i]

                # Check if next_node is within bounds and not an obstacle on the BUFFERED map
                if self.is_valid_position_on_map(next_node, self.buffered_floor_plan):
                    
                    current_direction_vector = (dr, dc)
                    
                    # Calculate turn penalty
                    turn_cost = 0
                    # Only apply turn penalty if there was a previous node to compare direction with
                    if parent_of_current is not None:
                        # Get the direction vector from the parent to the current node
                        prev_direction_vector = (current_node[0] - parent_of_current[0], current_node[1] - parent_of_current[1])
                        
                        # If the previous direction vector is not (0,0) (i.e., not the start node's parent)
                        # and the current movement direction is different from the previous one, apply penalty
                        if prev_direction_vector != (0,0) and current_direction_vector != prev_direction_vector:
                            turn_cost = turn_penalty
                    
                    new_g_cost = g_cost + move_cost + turn_cost

                    # If this path to next_node is better than any previous one
                    if next_node not in cost_so_far or new_g_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_g_cost
                        f_cost = new_g_cost + heuristic(next_node, goal)
                        # Push to open_set, storing current_node as the parent for next_node
                        heapq.heappush(open_set, (f_cost, new_g_cost, next_node, current_node))
                        came_from[next_node] = current_node
        
        # If the open set is empty and goal was not reached, no path exists
        return None

    # New helper method to find path from voice command
    def find_path_from_voice(self, start_coords, end_coords, end_name):
        if self.buffered_floor_plan is None:
            self.status_bar.showMessage("Please load a floor map and apply buffer first.")
            QMessageBox.warning(self, "Map Not Ready", "Please load a floor map and ensure the obstacle buffer is applied.")
            return

        # Check if start or end points are within the buffered obstacles
        if self.buffered_floor_plan[start_coords[0], start_coords[1]] == 1:
            QMessageBox.warning(self, "Invalid Start Point", "Start point is within a buffered obstacle. Please choose a free space.")
            self.status_bar.showMessage("Start point is in buffered obstacle.")
            return
        if self.buffered_floor_plan[end_coords[0], end_coords[1]] == 1:
            QMessageBox.warning(self, "Invalid End Point", "End point is within a buffered obstacle. Please choose a free space.")
            self.status_bar.showMessage("End point is in buffered obstacle.")
            return

        self.status_bar.showMessage(f"Finding path from voice command to {end_name}...")
        path = self.astar_path(start_coords, end_coords)

        if path is None:
            self.status_bar.showMessage(f"No path found to {end_name}.")
            QMessageBox.warning(self, "Path Not Found", f"No valid path could be found to {end_name}. Check for obstacles or increase buffer size.")
            return

        # Clear previous path line and animated dot
        if self.current_path_line:
            self.current_path_line.remove()
            self.current_path_line = None
        if self.animated_dot:
            self.animated_dot.remove()
            self.animated_dot = None
            if self.path_ax.legend_ is not None:
                self.path_ax.legend().remove()
                self.path_ax.legend([])

        # Redraw map, points, and names
        self.path_ax.clear()
        self.path_ax.imshow(self.buffered_floor_plan, cmap="gray")
        for pos in self.marked_positions:
            self.path_ax.plot(pos[1], pos[0], 'ro', markersize=8)
        for name, pos in self.marked_points.items():
            self.path_ax.text(pos[1], pos[0], name, color="blue", fontsize=10, ha='center', va='bottom')

        # Draw the new path
        path_r = [p[0] for p in path]
        path_c = [p[1] for p in path]
        self.current_path_line, = self.path_ax.plot(path_c, path_r, 'g-', linewidth=2, marker='o', markersize=4, markerfacecolor='green')
        
        # Start the animation
        self.current_animated_path = path
        self.current_path_index = 0
        self.path_animation_timer.start(100) # Animate every 100 ms (adjust speed as needed)

        self.status_bar.showMessage(f"Path found to {end_name} (length {len(path)} cells). Starting animation...")


    # ================== Static Image Detection tab ==================
    def setup_static_detection_tab(self):
        layout = QVBoxLayout(self.tab_static_detection)

        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)

        self.load_static_image_btn = QPushButton("Load Image")
        self.load_static_image_btn.clicked.connect(self.load_static_image)
        control_layout.addWidget(self.load_static_image_btn)

        self.detect_static_btn = QPushButton("Detect Objects (YOLOv5)")
        self.detect_static_btn.clicked.connect(lambda: self.detect_static_objects('yolov5'))
        control_layout.addWidget(self.detect_static_btn)

        self.detect_static_fasterrcnn_btn = QPushButton("Detect Objects (Faster R-CNN)")
        self.detect_static_fasterrcnn_btn.clicked.connect(lambda: self.detect_static_objects('fasterrcnn'))
        control_layout.addWidget(self.detect_static_fasterrcnn_btn)

        control_layout.addStretch(1)
        layout.addWidget(control_frame)

        self.static_image_label = QLabel("Loaded Image will appear here")
        self.static_image_label.setAlignment(Qt.AlignCenter)
        self.static_image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        layout.addWidget(self.static_image_label, 1)

        info_text = ("Upload a static image and use either YOLOv5 or Faster R-CNN to detect objects.\n"
                     "YOLOv5 is generally faster, Faster R-CNN can be more accurate for certain tasks.")
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.current_static_image = None # Store the PIL image for detection

    def load_static_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if not file_path:
            return
        try:
            # Load as PIL Image for model input (YOLOv5/Faster R-CNN expect PIL or Tensor)
            self.current_static_image = PIL.Image.open(file_path).convert("RGB")
            
            # Display the original image (without detections yet)
            pixmap = QPixmap(file_path)
            label_size = self.static_image_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.static_image_label.setPixmap(scaled_pixmap)
            self.status_bar.showMessage(f"Loaded image: {file_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading image: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")

    def detect_static_objects(self, model_type):
        if self.current_static_image is None:
            self.status_bar.showMessage("Please load an image first.")
            return

        # Convert PIL image to OpenCV BGR for drawing, this will be the image we modify
        # Ensure it's a copy so the original self.current_static_image (PIL) remains untouched
        img_np_rgb = np.array(self.current_static_image)
        img_cv_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

        num_detections = 0 # To count detections

        if model_type == 'yolov5':
            if yolo_model is None:
                self.status_bar.showMessage("YOLOv5 model not loaded.")
                QMessageBox.critical(self, "Model Error", "YOLOv5 model is not available.")
                return
            model_to_use = yolo_model
            class_names = model_to_use.names
            self.status_bar.showMessage("Detecting objects with YOLOv5...")
            print("Running YOLOv5 detection...")

            try:
                with torch.no_grad():
                    # YOLOv5 takes PIL image directly
                    results = model_to_use(self.current_static_image)
                detections = results.xyxy[0].cpu().numpy() # Get detections in numpy array

                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if conf > 0.5: # Confidence threshold
                        label = class_names[int(cls)]
                        # Draw bounding box (blue color in BGR)
                        cv2.rectangle(img_cv_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        # Put text (blue color in BGR)
                        cv2.putText(img_cv_bgr, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        num_detections += 1
                self.status_bar.showMessage(f"YOLOv5 detection complete. Found {num_detections} objects.")
                print(f"YOLOv5: Found {num_detections} objects.")

            except Exception as e:
                self.status_bar.showMessage(f"Error during YOLOv5 detection: {str(e)}")
                QMessageBox.critical(self, "Detection Error", f"An error occurred during YOLOv5 detection: {str(e)}")
                return # Exit if error

        elif model_type == 'fasterrcnn':
            if fasterrcnn_static_model is None:
                self.status_bar.showMessage("Faster R-CNN model not loaded.")
                QMessageBox.critical(self, "Model Error", "Faster R-CNN model is not available.")
                return
            model_to_use = fasterrcnn_static_model
            self.status_bar.showMessage("Detecting objects with Faster R-CNN...")
            print("Running Faster R-CNN detection...")

            try:
                # Faster R-CNN expects torch.Tensor
                img_tensor = F.to_tensor(self.current_static_image).to(device)
                with torch.no_grad():
                    prediction = model_to_use([img_tensor])

                # Process Faster R-CNN predictions
                boxes = prediction[0]['boxes'].cpu().numpy()
                labels = prediction[0]['labels'].cpu().numpy()
                scores = prediction[0]['scores'].cpu().numpy()

                for i in range(len(boxes)):
                    if scores[i] > 0.5: # Confidence threshold
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        label = get_coco_class_name(labels[i])
                        # Draw bounding box (red color in BGR)
                        cv2.rectangle(img_cv_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        # Put text (red color in BGR)
                        cv2.putText(img_cv_bgr, f"{label} {scores[i]:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        num_detections += 1
                self.status_bar.showMessage(f"Faster R-CNN detection complete. Found {num_detections} objects.")
                print(f"Faster R-CNN: Found {num_detections} objects.")

            except Exception as e:
                self.status_bar.showMessage(f"Error during Faster R-CNN detection: {str(e)}")
                QMessageBox.critical(self, "Detection Error", f"An error occurred during Faster R-CNN detection: {str(e)}")
                return # Exit if error
        else:
            self.status_bar.showMessage("Invalid model type specified.")
            return

        # Convert processed OpenCV image (BGR) back to QImage (BGR888) for display
        h, w, ch = img_cv_bgr.shape
        bytes_per_line = ch * w
        # Create a deep copy of the image data to ensure QImage retains it
        q_img = QImage(img_cv_bgr.copy().data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        label_size = self.static_image_label.size()
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.static_image_label.setPixmap(scaled_pixmap)

    # ================== Video Detection tab ==================
    class VideoProcessingWorker(QThread):
        image_update = pyqtSignal(QImage)
        status_update = pyqtSignal(str)

        def __init__(self, model, model_type):
            super().__init__()
            self._run_flag = True
            self.model = model
            self.model_type = model_type
            self.cap = None
            self.video_path = None

        def set_video_path(self, path):
            self.video_path = path

        def run(self):
            if self.model is None:
                self.status_update.emit(f"Error: {self.model_type} model not loaded.")
                self._run_flag = False
                return
            if self.video_path is None:
                self.status_update.emit("Error: No video path set.")
                self._run_flag = False
                return

            try:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    self.status_update.emit(f"Error: Could not open video file: {self.video_path}. Check file path and codecs.")
                    print(f"Error: cv2.VideoCapture failed to open {self.video_path}")
                    self._run_flag = False
                    return
                print(f"Successfully opened video: {self.video_path}")
            except Exception as e:
                self.status_update.emit(f"Critical error opening video: {e}")
                print(f"Critical error opening video: {e}")
                self._run_flag = False
                return

            self.status_update.emit(f"Processing video with {self.model_type}...")

            while self._run_flag:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_update.emit("End of video or error reading frame. Stopping video processing.")
                    print("End of video or failed to read frame.")
                    break # End of video or error
                
                # Convert frame to RGB for model input
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    if self.model_type == 'yolov5':
                        with torch.no_grad():
                            results = self.model(frame_rgb)
                        detections = results.xyxy[0].cpu().numpy()

                        for det in detections:
                            x1, y1, x2, y2, conf, cls = det
                            if conf > 0.5:
                                label = self.model.names[int(cls)]
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif self.model_type == 'fasterrcnn':
                        img_pil = PIL.Image.fromarray(frame_rgb)
                        img_tensor = F.to_tensor(img_pil).to(device)
                        with torch.no_grad():
                            prediction = self.model([img_tensor])

                        boxes = prediction[0]['boxes'].cpu().numpy()
                        labels = prediction[0]['labels'].cpu().numpy()
                        scores = prediction[0]['scores'].cpu().numpy()

                        for i in range(len(boxes)):
                            if scores[i] > 0.5:
                                x1, y1, x2, y2 = boxes[i].astype(int)
                                label = get_coco_class_name(labels[i])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f"{label} {scores[i]:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except Exception as e:
                    self.status_update.emit(f"Error during video inference: {e}. Stopping video processing.")
                    print(f"Error during video inference: {e}")
                    self._run_flag = False
                    break # Stop processing on error

                h, w, ch = frame.shape
                bytes_per_line = ch * w
                # Create a deep copy of the image data to ensure QImage retains it
                q_img = QImage(frame.copy().data, w, h, bytes_per_line, QImage.Format_BGR888)
                self.image_update.emit(q_img)
                
                # Control frame rate for smoother playback
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    time.sleep(1 / fps)
                else:
                    time.sleep(0.03) # Default sleep if FPS is not available or zero

            self.cap.release()
            self.status_update.emit("Video processing finished.")
            self._run_flag = False # Ensure flag is reset after loop completion

        def stop(self):
            self._run_flag = False
            self.wait() # Wait for the thread to finish

    def setup_video_detection_tab(self):
        layout = QVBoxLayout(self.tab_video_detection)

        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)

        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video_file)
        control_layout.addWidget(self.load_video_btn)

        self.start_video_yolo_btn = QPushButton("Start Detection (YOLOv5)")
        self.start_video_yolo_btn.clicked.connect(lambda: self.start_video_detection('yolov5'))
        self.start_video_yolo_btn.setEnabled(False)
        control_layout.addWidget(self.start_video_yolo_btn)

        self.start_video_fasterrcnn_btn = QPushButton("Start Detection (Faster R-CNN)")
        self.start_video_fasterrcnn_btn.clicked.connect(lambda: self.start_video_detection('fasterrcnn'))
        self.start_video_fasterrcnn_btn.setEnabled(False)
        control_layout.addWidget(self.start_video_fasterrcnn_btn)

        self.stop_video_btn = QPushButton("Stop Video")
        self.stop_video_btn.clicked.connect(self.stop_video_detection)
        self.stop_video_btn.setEnabled(False)
        control_layout.addWidget(self.stop_video_btn)

        control_layout.addStretch(1)
        layout.addWidget(control_frame)

        self.video_display_label = QLabel("Video Feed will appear here")
        self.video_display_label.setAlignment(Qt.AlignCenter)
        self.video_display_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        layout.addWidget(self.video_display_label, 1)

        info_text = ("Load a video file and perform object detection using either YOLOv5 or Faster R-CNN.\n"
                     "Note: Video processing can be resource-intensive.")
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.current_video_path = None

    def load_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if not file_path:
            self.status_bar.showMessage("Video file selection cancelled.")
            return
        self.current_video_path = file_path
        self.status_bar.showMessage(f"Video loaded: {file_path}")
        print(f"Video file selected: {self.current_video_path}")
        self.start_video_yolo_btn.setEnabled(True)
        self.start_video_fasterrcnn_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(False) # Ensure stop is disabled until started

        # Clear previous video frame
        self.video_display_label.clear()

    def start_video_detection(self, model_type):
        if self.current_video_path is None:
            self.status_bar.showMessage("Please load a video file first.")
            return
        
        if self.video_processing_thread and self.video_processing_thread.isRunning():
            self.status_bar.showMessage("Video processing already running.")
            return

        model_to_use = None
        if model_type == 'yolov5':
            model_to_use = yolo_model
            if model_to_use is None:
                QMessageBox.critical(self, "Model Error", "YOLOv5 model is not available.")
                self.status_bar.showMessage("YOLOv5 model not loaded.")
                return
        elif model_type == 'fasterrcnn':
            model_to_use = fasterrcnn_video_model # Use the video-optimized Faster R-CNN
            if model_to_use is None:
                QMessageBox.critical(self, "Model Error", "Faster R-CNN model is not available.")
                self.status_bar.showMessage("Faster R-CNN model not loaded.")
                return
        else:
            self.status_bar.showMessage("Invalid model type specified.")
            return

        self.video_processing_thread = self.VideoProcessingWorker(model_to_use, model_type)
        self.video_processing_thread.set_video_path(self.current_video_path)
        self.video_processing_thread.image_update.connect(self.update_video_frame)
        self.video_processing_thread.status_update.connect(self.status_bar.showMessage)
        
        self.start_video_yolo_btn.setEnabled(False)
        self.start_video_fasterrcnn_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(True)
        self.video_processing_thread.start()
        print(f"Starting video detection thread for {model_type}...")

    def stop_video_detection(self):
        if self.video_processing_thread and self.video_processing_thread.isRunning():
            self.video_processing_thread.stop()
            self.video_processing_thread.wait()
            self.status_bar.showMessage("Video processing stopped.")
            print("Video detection thread stopped.")
        self.start_video_yolo_btn.setEnabled(True)
        self.start_video_fasterrcnn_btn.setEnabled(True)
        self.stop_video_btn.setEnabled(False)
        self.video_display_label.clear()

    @QtCore.pyqtSlot(QImage)
    def update_video_frame(self, image):
        label_size = self.video_display_label.size()
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_display_label.setPixmap(scaled_pixmap)

    # ================== Voice Control tab ==================
    class VoiceControlWorker(QThread):
        command_detected = pyqtSignal(str)
        status_update = pyqtSignal(str)
        
        def __init__(self):
            super().__init__()
            self._run_flag = True
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

        def run(self):
            self.status_update.emit("Listening for commands...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source) # Adjust for ambient noise once

                while self._run_flag:
                    try:
                        self.status_update.emit("Say something!")
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                        self.status_update.emit("Recognizing...")
                        command = self.recognizer.recognize_google(audio).lower()
                        self.status_update.emit(f"Heard: {command}")
                        self.command_detected.emit(command)
                    except sr.WaitTimeoutError:
                        self.status_update.emit("No speech detected. Listening again...")
                        continue
                    except sr.UnknownValueError:
                        self.status_update.emit("Could not understand audio. Listening again...")
                        continue
                    except sr.RequestError as e:
                        self.status_update.emit(f"Could not request results from Google Speech Recognition service; {e}. Retrying...")
                        time.sleep(1) # Wait before retrying
                        continue
                    except Exception as e:
                        self.status_update.emit(f"An unexpected error occurred: {e}. Stopping voice control.")
                        self._run_flag = False
            self.status_update.emit("Voice control stopped.")

        def stop(self):
            self._run_flag = False
            self.wait()

    def setup_voice_control_tab(self):
        layout = QVBoxLayout(self.tab_voice_control)

        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)

        self.load_voice_map_btn = QPushButton("Load Floor Map")
        self.load_voice_map_btn.clicked.connect(self.load_voice_map)
        control_layout.addWidget(self.load_voice_map_btn)

        self.start_voice_btn = QPushButton("Start Voice Control")
        self.start_voice_btn.clicked.connect(self.start_voice_control)
        control_layout.addWidget(self.start_voice_btn)

        self.stop_voice_btn = QPushButton("Stop Voice Control")
        self.stop_voice_btn.clicked.connect(self.stop_voice_control)
        self.stop_voice_btn.setEnabled(False)
        control_layout.addWidget(self.stop_voice_btn)

        control_layout.addStretch(1)
        layout.addWidget(control_frame)

        # Frame for the Matplotlib voice control map
        self.voice_map_frame = QWidget()
        voice_map_layout = QVBoxLayout(self.voice_map_frame)
        layout.addWidget(self.voice_map_frame, 1)

        self.voice_map_fig = Figure(figsize=(6,6), dpi=100)
        self.voice_map_ax = self.voice_map_fig.add_subplot(111)
        self.voice_map_canvas = FigureCanvas(self.voice_map_fig)
        voice_map_layout.addWidget(self.voice_map_canvas)

        # Voice command input and send button
        command_input_frame = QWidget()
        command_input_layout = QHBoxLayout(command_input_frame)
        command_input_layout.addWidget(QLabel("Type Command:"))
        self.voice_command_input = QLineEdit()
        self.voice_command_input.returnPressed.connect(self.send_typed_command) # Connect Enter key
        command_input_layout.addWidget(self.voice_command_input)
        self.send_voice_command_btn = QPushButton("Send")
        self.send_voice_command_btn.clicked.connect(self.send_typed_command)
        command_input_layout.addWidget(self.send_voice_command_btn)
        layout.addWidget(command_input_frame)


        self.voice_command_label = QLabel("Last Command: None")
        self.voice_command_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.voice_command_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.voice_command_label)

        self.voice_action_label = QLabel("Action: Waiting...")
        self.voice_action_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.voice_action_label)

        info_text = ("Use voice commands to control the wheelchair simulation.\n"
                     "Supported commands: 'move forward', 'move backward', 'turn left', 'turn right', 'stop'.\n"
                     "For Path Planning: 'go to [point name]'.\n"
                     "You can also type commands if microphone is not working.")
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

    def load_voice_map(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Floor Map Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if not file_path:
            return
        try:
            im = PIL.Image.open(file_path).convert("L") # Convert to grayscale
            self.voice_map = np.array(im)
            # Convert to binary map: obstacles=1 (dark), free=0 (light)
            self.voice_map = np.where(self.voice_map < 128, 1, 0)
            self.voice_map_ax.clear()
            self.voice_map_ax.imshow(self.voice_map, cmap="gray")

            # Stop any ongoing movement when a new map is loaded
            self.voice_move_timer.stop()
            self.voice_current_direction = None
            if self.voice_dot:
                self.voice_dot.remove() # Remove old dot
                self.voice_dot = None
                if self.voice_map_ax.legend_ is not None:
                    self.voice_map_ax.legend().remove()
                    self.voice_map_ax.legend([])


            # Set initial position to center, then find closest valid free space
            rows, cols = self.voice_map.shape
            initial_r, initial_c = rows // 2, cols // 2
            self.voice_map_position = (initial_r, initial_c)

            if self.voice_map[self.voice_map_position] != 0: # If center is an obstacle
                found = False
                # Search outwards in increasing radii for a free spot
                for radius in range(max(rows, cols)):
                    for dr in range(-radius, radius+1):
                        for dc in range(-radius, radius+1):
                            r = initial_r + dr
                            c = initial_c + dc
                            if 0 <= r < rows and 0 <= c < cols: # Check bounds
                                if self.voice_map[r, c] == 0: # Found a free spot
                                    self.voice_map_position = (r, c)
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    QMessageBox.warning(self, "Map Error", "Could not find a free starting position on the map.")
                    self.status_bar.showMessage("Error: No free starting position found on voice map.")
                    return

            # Plot the current position dot
            self.voice_dot, = self.voice_map_ax.plot(self.voice_map_position[1], self.voice_map_position[0], 'yo', markersize=8, label='Wheelchair') # Yellow dot
            self.voice_map_ax.legend()
            self.voice_map_canvas.draw() # Redraw the canvas
            self.status_bar.showMessage(f"Loaded voice control map: {file_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error loading voice control map: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error loading voice control map: {str(e)}")

    def move_voice_map(self, direction):
        """
        Attempts to move the wheelchair one step in the given direction on the voice control map.
        Returns True if moved successfully, False otherwise (hit obstacle/wall).
        """
        if self.voice_map is None or self.voice_map_position is None:
            return False

        r, c = self.voice_map_position
        new_pos = (r, c)

        # Calculate new position based on direction
        if direction == 'up':
            new_pos = (r-1, c)
        elif direction == 'down':
            new_pos = (r+1, c)
        elif direction == 'left':
            new_pos = (r, c-1)
        elif direction == 'right':
            new_pos = (r, c+1)
        else:
            return False # Invalid direction

        # Check if the new position is valid (within bounds and not an obstacle)
        if self.is_valid_position_on_map(new_pos, self.voice_map):
            self.voice_map_position = new_pos
            self.voice_dot.set_data([new_pos[1]], [new_pos[0]]) # Update dot's coordinates
            self.voice_map_canvas.draw_idle() # Use draw_idle for continuous animation efficiency
            return True
        else:
            return False # Move blocked by obstacle or wall

    def animate_voice_move_step(self):
        """
        Called by the voice_move_timer to continuously move the wheelchair on the voice map.
        Stops the timer if an obstacle or wall is hit.
        """
        if self.voice_current_direction is None:
            self.voice_move_timer.stop()
            self.status_bar.showMessage("Voice movement stopped.")
            return

        moved_successfully = self.move_voice_map(self.voice_current_direction)
        if not moved_successfully:
            self.voice_move_timer.stop()
            self.voice_current_direction = None
            self.status_bar.showMessage("Voice movement stopped: Hit obstacle or wall.")
            self.voice_action_label.setText("Action: Stopped (Obstacle)")
        else:
            self.status_bar.showMessage(f"Moving {self.voice_current_direction} on voice map.")


    def send_typed_command(self):
        command = self.voice_command_input.text().strip().lower()
        if command:
            self.process_voice_command(command)
            self.voice_command_input.clear() # Clear the input field

    def start_voice_control(self):
        if self.voice_control_thread and self.voice_control_thread.isRunning():
            return

        self.voice_control_thread = self.VoiceControlWorker()
        self.voice_control_thread.command_detected.connect(self.process_voice_command)
        self.voice_control_thread.status_update.connect(self.status_bar.showMessage)
        
        self.start_voice_btn.setEnabled(False)
        self.stop_voice_btn.setEnabled(True)
        self.voice_control_thread.start()
        self.voice_command_label.setText("Last Command: None")
        self.voice_action_label.setText("Action: Waiting...")

    def stop_voice_control(self):
        if self.voice_control_thread and self.voice_control_thread.isRunning():
            self.voice_control_thread.stop()
            self.voice_control_thread.wait()
            self.status_bar.showMessage("Voice control stopped.")
        # Also stop the voice map movement timer if it's running
        self.voice_move_timer.stop()
        self.voice_current_direction = None
        self.start_voice_btn.setEnabled(True)
        self.stop_voice_btn.setEnabled(False)
        self.voice_command_label.setText("Last Command: None")
        self.voice_action_label.setText("Action: Stopped.")

    @QtCore.pyqtSlot(str)
    def process_voice_command(self, command):
        self.voice_command_label.setText(f"Last Command: {command}")
        action = "Unknown command"
        
        current_tab_index = self.notebook.currentIndex()
        
        if current_tab_index == self.notebook.indexOf(self.tab_manual_navigation):
            # Handle manual navigation commands
            if "move forward" in command:
                action = "Moving Forward (Manual)"
                self.move_manual('up') # Assuming 'up' is forward on map
            elif "move backward" in command:
                action = "Moving Backward (Manual)"
                self.move_manual('down')
            elif "turn left" in command:
                action = "Turning Left (Manual)"
                self.move_manual('left')
            elif "turn right" in command:
                action = "Turning Right (Manual)"
                self.move_manual('right')
            elif "stop" in command:
                action = "Stopping (Manual)"
                # For manual control, stopping means no further movement initiated by voice
                # No direct "stop" function for manual movement, just don't call move_manual
            self.voice_action_label.setText(f"Action: {action}")

        elif current_tab_index == self.notebook.indexOf(self.tab_path_planning):
            # Handle path planning commands
            match = re.search(r"go to (.*)", command)
            if match:
                target_point_name = match.group(1).strip()
                if target_point_name in self.marked_points:
                    action = f"Planning path to {target_point_name}"
                    
                    start_point = None
                    if self.animated_dot and self.current_animated_path and self.current_path_index > 0 and self.current_path_index <= len(self.current_animated_path):
                        # If animation is active, start from current animated position (last drawn point)
                        start_point = self.current_animated_path[self.current_path_index - 1] 
                    elif self.marked_positions:
                        # If no animation, but points are marked, use the first marked point as start
                        start_point = self.marked_positions[0]
                    else:
                        self.status_bar.showMessage("Cannot plan path: No current position or marked points on map.")
                        action = "Error: No start point for path planning."
                        self.voice_action_label.setText(f"Action: {action}")
                        return

                    if start_point:
                        end_point = self.marked_points[target_point_name]
                        self.find_path_from_voice(start_point, end_point, target_point_name)
                    else:
                        self.status_bar.showMessage("No valid start point for path planning from voice command.")
                        action = "Error: No start point."
                else:
                    action = f"Point '{target_point_name}' not found."
            elif "stop" in command:
                action = "Stopping Path Animation"
                self.path_animation_timer.stop()
                self.status_bar.showMessage("Path animation paused.")
            else:
                action = "Unknown command for Path Planning tab."
            self.voice_action_label.setText(f"Action: {action}")

        elif current_tab_index == self.notebook.indexOf(self.tab_voice_control):
            # Handle voice control commands for its own map
            if self.voice_map is None or self.voice_map_position is None:
                self.status_bar.showMessage("Please load a floor map in Voice Control tab first.")
                action = "Error: No map loaded for voice control."
                self.voice_action_label.setText(f"Action: {action}")
                return

            if "move forward" in command:
                action = "Moving Forward (Voice Map)"
                self.voice_current_direction = 'up'
                self.voice_move_timer.start(100) # Start continuous movement
            elif "move backward" in command:
                action = "Moving Backward (Voice Map)"
                self.voice_current_direction = 'down'
                self.voice_move_timer.start(100)
            elif "turn left" in command:
                action = "Turning Left (Voice Map)"
                self.voice_current_direction = 'left'
                self.voice_move_timer.start(100)
            elif "turn right" in command:
                action = "Turning Right (Voice Map)"
                self.voice_current_direction = 'right'
                self.voice_move_timer.start(100)
            elif "stop" in command:
                action = "Stopping (Voice Map)"
                self.voice_move_timer.stop()
                self.voice_current_direction = None # Clear current direction
            else:
                action = "Unknown command for Voice Control map."
            self.voice_action_label.setText(f"Action: {action}")
        else:
            action = "Unknown command or tab not active."
            self.voice_action_label.setText(f"Action: {action}")

        self.status_bar.showMessage(f"Processed voice command: {command}")

    # --- Application Cleanup ---
    def closeEvent(self, event):
        # Ensure all threads are stopped when the main window closes
        if self.camera_worker_thread and self.camera_worker_thread.isRunning():
            self.camera_worker_thread.stop()
            self.camera_worker_thread.wait()
        if self.video_processing_thread and self.video_processing_thread.isRunning():
            self.video_processing_thread.stop()
            self.video_processing_thread.wait()
        if self.voice_control_thread and self.voice_control_thread.isRunning():
            self.voice_control_thread.stop()
            self.voice_control_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WheelchairSimulationApp()
    window.show()
    sys.exit(app.exec_())
