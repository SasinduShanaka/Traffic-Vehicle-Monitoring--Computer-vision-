from flask import Flask, render_template, request, send_from_directory
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO("model/yolov8n.pt")

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


@app.route("/", methods=["GET", "POST"])
def index():
    output_video = None
    vehicle_counts = None
    traffic_level = None

    if request.method == "POST":
        file = request.files.get("video")

        if file and file.filename != "":
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            output_filename = "output_" + os.path.splitext(file.filename)[0] + ".mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            file.save(input_path)

            vehicle_counts, traffic_level = process_video(input_path, output_path)
            output_video = output_filename

    return render_template(
        "index.html",
        output_video=output_video,
        vehicle_counts=vehicle_counts,
        traffic_level=traffic_level
    )


@app.route("/outputs/<filename>")
def output_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Store unique vehicle IDs detected
    counted_ids = {
        "Car": set(),
        "Motorcycle": set(),
        "Bus": set(),
        "Truck": set()
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tracking with confidence threshold
        results = model.track(frame, persist=True, conf=0.25, iou=0.5)
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                cls_id = int(box.cls[0])
                track_id = int(track_id)
                conf = float(box.conf[0])
                
                # Only count vehicles with good confidence
                if conf < 0.25:
                    continue

                if cls_id in VEHICLE_CLASSES:
                    vehicle_name = VEHICLE_CLASSES[cls_id]
                    # Add to counted IDs (automatically handles uniqueness with set)
                    counted_ids[vehicle_name].add(track_id)

        # Display current counts on frame
        y_offset = 30
        for vehicle_type, ids in counted_ids.items():
            count_text = f"{vehicle_type}: {len(ids)}"
            cv2.putText(annotated_frame, count_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Final counts
    counts = {
        "Car": len(counted_ids["Car"]),
        "Motorcycle": len(counted_ids["Motorcycle"]),
        "Bus": len(counted_ids["Bus"]),
        "Truck": len(counted_ids["Truck"])
    }

    total = sum(counts.values())

    if total < 10:
        traffic_level = "LOW"
    elif total <= 20:
        traffic_level = "MEDIUM"
    else:
        traffic_level = "HIGH"

    return counts, traffic_level



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
