import sqlite3
from ultralytics import YOLO
from datetime import datetime
import os

# === CONFIG ===
IMAGE_FOLDER = "images"   # folder containing traffic images
INTERSECTION_ID = 1       # fixed intersection

# Load YOLO model
model = YOLO("yolov8n.pt")  # lightweight model

# Connect to database
conn = sqlite3.connect("traffic.sqlite")
cursor = conn.cursor()

# Classes we consider as vehicles
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# Loop through images
for image_name in os.listdir(IMAGE_FOLDER):
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    print(f"Processing: {image_name}")

    results = model(image_path)

    vehicle_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in VEHICLE_CLASSES:
                vehicle_count += 1

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert into database
    cursor.execute("""
        INSERT INTO detections (intersection_id, detection_timestamp, vehicle_count)
        VALUES (?, ?, ?)
    """, (INTERSECTION_ID, timestamp, vehicle_count))

    print(f"Detected {vehicle_count} vehicles")

# Save changes
conn.commit()
conn.close()

print("✅ All images processed and stored in database.")