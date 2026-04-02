import sqlite3
from datetime import timedelta, datetime
import random
import time

conn = sqlite3.connect("traffic.sqlite")
cursor = conn.cursor()

base_time = datetime.now()
for i in range(10):
    intersection_id = random.randint(1, 4)
    timestamp = (base_time + timedelta(minutes=i * 10)).strftime("%Y-%m-%d %H:%M:%S")
    vehicle_count = random.randint(1, 20)

    cursor.execute(
        """
    INSERT INTO detections (intersection_id, detection_timestamp, vehicle_count)
    VALUES (?, ?, ?)
    """,
        (intersection_id, timestamp, vehicle_count),
    )

    print(f"Inserted detection {i+1}: {vehicle_count} vehicles")

    time.sleep(1)

conn.commit()
conn.close()

print("✅ Simulation complete!")
