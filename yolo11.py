from ultralytics import YOLO
import cv2
import time
import numpy as np
import math
import torch
import urllib.request

# Download MiDaS model
'''midas_model_url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt"
midas_model_path = "midas_v21_small.pt"
urllib.request.urlretrieve(midas_model_url, midas_model_path)'''

# Load MiDaS
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Load the YOLO model
model = YOLO('model_- 20 october 2024 23_48.pt')

# Open the video file
cap = cv2.VideoCapture("output_video.mp4")

if not cap.isOpened():
    raise IOError("Cannot open video file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('payload_with_depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Define colors for the 4 triangular faces
FACE_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0)   # Yellow
]

def calculate_object_metrics(points, depth_map):
    """Calculate comprehensive metrics for the entire object"""
    # Extract just the x and y coordinates from each point
    pts = np.array([[p[0], p[1]] for p in points], dtype=np.float32).reshape((-1, 1, 2))
    
    # Basic measurements
    area = cv2.contourArea(pts)
    perimeter = cv2.arcLength(pts, True)
    
    # Calculate bounding box
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Calculate aspect ratio
    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[1] - box[2])
    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)  # avoid division by zero
    
    # Calculate rotation based on the front face (first three points)
    front_face = points[:3]
    apex = front_face[0][:2]  # Only take x and y coordinates
    base_mid = ((front_face[1][:2] + front_face[2][:2]) / 2).astype(int)  # Only take x and y coordinates
    
    # Calculate the vector from base midpoint to apex
    direction_vector = apex - base_mid
    
    # Calculate the angle of this vector with respect to the vertical (y-axis)
    angle = math.degrees(math.atan2(direction_vector[0], -direction_vector[1]))
    
    # Normalize angle to be between 0 and 360 degrees
    rotation = (angle + 360) % 360
    
    # Calculate the object's centroid
    M = cv2.moments(pts)
    centroid_x = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
    centroid_y = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
    
    # Calculate average depth of the object
    object_mask = np.zeros(depth_map.shape, dtype=np.uint8)
    cv2.fillPoly(object_mask, [pts.astype(int)], 255)
    avg_depth = np.mean(depth_map[object_mask == 255])

    return {
        'area': area,
        'perimeter': perimeter,
        'rotation': rotation,
        'aspect_ratio': aspect_ratio,
        'centroid': (centroid_x, centroid_y),
        'apex': apex,
        'base_mid': base_mid,
        'avg_depth': avg_depth
    }

try:
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO inference
        results = model(frame)
        
        # Run MiDaS inference
        input_batch = transform(frame).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Process each detection
        # Process each detection
        if len(results[0].boxes.data) > 0:
            for box, keypoints in zip(results[0].boxes.data, results[0].keypoints.data):
                # Draw bounding box
                x1, y1, x2, y2, conf, cls = box
                cv2.rectangle(annotated_frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Convert keypoints to integer coordinates
                kpts = keypoints.cpu().numpy().astype(np.int32)
                
                # Draw keypoints
                for point in kpts:
                    cv2.circle(annotated_frame, (point[0], point[1]), 5, (0, 255, 0), -1)
                
                # Define the 4 triangular faces (apex to base points)
                faces = [
                    [kpts[0], kpts[1], kpts[2]],  # Front face
                    [kpts[0], kpts[2], kpts[3]],  # Right face
                    [kpts[0], kpts[3], kpts[4]],  # Back face
                    [kpts[0], kpts[4], kpts[1]]   # Left face
                ]
                
                # Draw each triangular face
                for i, face_points in enumerate(faces):
                    pts = np.array([[[p[0], p[1]] for p in face_points]], dtype=np.int32)
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, pts, FACE_COLORS[i])
                    cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                
                # Calculate metrics for the entire object
                all_points = np.vstack(faces)
                metrics = calculate_object_metrics(all_points, depth_map)
                
                # Display measurements
                measurements_y = 60  # Starting y position for text
                cv2.putText(annotated_frame, "Object Measurements:", (10, measurements_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                measurements_y += 25
                
                # Format measurements text
                measurements_text = [
                    f"Area: {metrics['area']:.1f} px²",
                    f"Perimeter: {metrics['perimeter']:.1f} px",
                    f"Rotation: {metrics['rotation']:.1f}°",
                    f"Aspect Ratio: {metrics['aspect_ratio']:.2f}",
                    f"Centroid: ({metrics['centroid'][0]}, {metrics['centroid'][1]})",
                    f"Avg Depth: {metrics['avg_depth']:.2f}"
                ]
                
                # Display all measurements
                for j, text in enumerate(measurements_text):
                    cv2.putText(annotated_frame, text, 
                            (10, measurements_y + j*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                apex = metrics['apex']
                base_mid = metrics['base_mid']
                cv2.line(annotated_frame, tuple(base_mid), tuple(apex.astype(int)), (0, 255, 255), 2)
        
        # Overlay depth map on the frame
        depth_colormap = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, depth_colormap, 0.3, 0)
        
        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(
            annotated_frame,
            f'FPS: {fps:.2f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Save and display the frame
        out.write(annotated_frame)
        cv2.imshow('Pyramid Object Detection with Depth', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()