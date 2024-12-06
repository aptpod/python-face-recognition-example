import face_recognition
import cv2
import sys
import numpy as np
import os
import time

if len(sys.argv) != 2:
    print("Please provide the path to the video file")
    sys.exit(1)

video_capture = cv2.VideoCapture(sys.argv[1])
if not video_capture.isOpened():
    print("Error opening video file")
    sys.exit(1)

fps = video_capture.get(cv2.CAP_PROP_FPS)
print("FPS: {}".format(fps))

output_dir = int(time.time())
face_groups = [] # list of face encodings
frame_count = 0

while video_capture.isOpened():
    ret, raw_frame = video_capture.read()
    if not ret:
        print("Video file ended")
        break

    frame_count += 1
    # take a frame every 2 seconds
    if frame_count % int(fps*2) != 0:
        continue

      # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frame)

    frame_time = time.gmtime(video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    print("Found {} face(s) in frame #{}, time {}.".format(len(face_locations), frame_count, time.strftime("%H:%M:%S", frame_time)))
    if len(face_locations) == 0:
        continue

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    for i in range(len(face_locations)):
        face_location = face_locations[i]
        face_encoding = face_encodings[i]

        # Print the location of each face in this frame
        top, right, bottom, left = face_location
        face_size = (right - left) * (bottom - top)
        print(" - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}, Size: {}".format(top, left, bottom, right, face_size))

        group_id = -1
        if len(face_groups) > 0:
          distances = face_recognition.face_distance(face_groups, face_encoding)
          print("  - Min distance: {}".format(np.min(distances)))
          if np.min(distances) < 0.6: # threshold to consider the face as the same person
              group_id = np.argmin(distances)

        if group_id == -1: # create a new group
            face_groups.append(face_encoding)
            group_id = len(face_groups) - 1

        # save image
        name = "./output/{}/{}/{}.jpg".format(output_dir, group_id, time.strftime("%H_%M_%S", frame_time))
        os.makedirs(os.path.dirname(name), exist_ok=True)
        print("  - Saving face to {}".format(name))
        face = raw_frame[top:bottom, left:right]
        if cv2.imwrite(name, face, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) == False:
            print("  - Error saving face to {}".format(name))
