import cv2
import numpy as np
import time

# Initialize CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Take camera feed from default camera 0 -> use IP address of camera
video = cv2.VideoCapture(0)

time.sleep(1) # Allow camera to set up
success, frame = video.read() # Take current frame from video

img_width = frame.shape[1]
img_height = frame.shape[0]

hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv_image)

# Define color ranges for red and green in HSV
# Define the lower and upper HSV values for red
lower_red = np.array([0, 120, 140])
upper_red = np.array([10, 200, 210])

# Create a mask for red pixels using cv2.inRange
mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

# Define the lower and upper HSV values for green
lower_green = np.array([60, 60, 50])
upper_green = np.array([90, 170, 255])

# Create a mask for green pixels using cv2.inRange
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
cv2.imshow('Red', mask_red)
cv2.imshow('Green',mask_green)

# Combine the masks to select region with both red and green
combined_mask = cv2.bitwise_and(~mask_green,~mask_red)
cv2.imshow('Mask',combined_mask)
result = cv2.bitwise_and(frame, frame, mask=~combined_mask)
cv2.imshow('Result',result)

# Remove background noise
blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

# Apply Canny edge detector
edges = cv2.Canny(blurred, 50, 150)
edges_with_mask = cv2.bitwise_and(edges, mask_red)
cv2.imshow('Edges with mask', edges_with_mask)

# Define a kernel for dilation
kernel = np.ones((8, 8), np.uint8)

# Perform dilation on the edges
dilated_edges = cv2.dilate(edges_with_mask, kernel, iterations=1)

cv2.imshow('Original Image', frame)
cv2.imshow('Canny Edges', edges)
cv2.imshow('Dilated Edges', dilated_edges)

# Find contours in the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour (assuming it corresponds to the desired region)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Draw a rectangle around the selected ROI
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Original Frame', frame)

    roi = frame[y:y + h, x:x + w]

    bbox = (x,y,w,h)

    # Initialize tracker with the initial bounding box
    tracker.init(frame, bbox)
else:
    bbox = cv2.selectROI('Frame',frame) # If program is unable to identify ROI, manually select

tracker.init(frame, bbox)  # Initialize tracker with the initial bounding box

while True:
    success, frame = video.read()
    if not success:
        break

    success, bbox = tracker.update(frame)  # Update the tracker

    if success:
        # Get ROI box coordinates
        (x, y, w, h) = [int(i) for i in bbox]

        # If bounding box starts to go too far from the center(using 10% margin as 'too far' for now), camera position needs to be changed(changing the box color as a sign for now)
        if x < 0.1*img_width or y < 0.1*img_height or x + w > 0.9*img_width or y + h > 0.9*img_height:
            red = 255 # Box turns blue when it enters 10% margin
        else:
            red = 0 # Else, box will be green

        # Draw bounding box on the object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (red, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Breaks if you press 'q'
        break

video.release()
cv2.destroyAllWindows()

'''
The CSRT algorithm employs a discriminative correlation filter to estimate the object’s location and appearance.
This filter learns the object’s appearance using positive and negative training samples and is updated iteratively
during tracking.
https://medium.com/@khwabkalra1/object-tracking-2fe4127e58bf#:~:text=Mathematical%20Basis%3A%20The%20CSRT%20algorithm,is%20updated%20iteratively%20during%20tracking.'''

#bbox = cv2.selectROI("Frame", frame, False)  # select the object to track(manual selection). color segmentation?
