import cv2
from ultralytics import YOLO
import collections

# 1. Load the pre-trained YOLO model
model = YOLO('yolov8n.pt')

# 2. Start the camera capture
cap = cv2.VideoCapture(0)

# 3. Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream. Please check your camera permissions.")
    exit()

# List to store all detected objects
detected_objects = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Run the YOLO model on the current frame
    results = model(frame, verbose=False)

    # 5. Get the detected objects and add to the list
    if results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detected_objects.append((class_name, confidence))

    # 6. Display the results on the frame
    annotated_frame = results[0].plot()
    
    # 7. Show the window
    cv2.imshow('YOLO Human/Animal Detection', annotated_frame)

    # 8. Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()

# 10. Print the final list of detected objects
if detected_objects:
    # Use collections.Counter to count occurrences and get the most common ones
    counts = collections.Counter([obj[0] for obj in detected_objects])
    
    print("--- Session Summary ---")
    print("Total unique objects detected:", len(counts))
    print("Most commonly detected objects:")
    for item, count in counts.most_common(5):
        print(f"- {item}: {count} times")

    print("\n--- All Detections ---")
    # You can choose to print all detections or just a summary
    # Here, we print a sample of the first 20 detections
    for i, (name, conf) in enumerate(detected_objects[:20]):
        print(f"Detected: {name} with {conf:.2f} confidence")
    if len(detected_objects) > 20:
        print("...(and more)")

else:
    print("No objects were detected during the session.")