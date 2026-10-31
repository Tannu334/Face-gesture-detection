
import cv2
import mediapipe as mp
import time

# -------------------------------
# INITIALIZATION
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load face detector")

# Finger landmark IDs
TIPS = [8, 12, 16, 20]
PIPS = [6, 10, 14, 18]


# -------------------------------
# FINGER COUNTING
# -------------------------------
def analyze_hand(hand_landmarks, handedness):
    lm = hand_landmarks.landmark

    # Check thumb extension.
    # MediaPipe handedness assumes mirrored selfie input.
    if handedness == "Right":
        thumb_open = lm[4].x < lm[3].x
    else:
        thumb_open = lm[4].x > lm[3].x

    # Check four fingers
    extended = []

    for tip, pip in zip(TIPS, PIPS):
        extended.append(lm[tip].y < lm[pip].y)

    finger_count = sum(extended) + int(thumb_open)

    # Identify a thumb-up gesture:
    # thumb extended upwards, other four fingers folded.
    thumb_up = (
        lm[4].y < lm[3].y
        and not any(extended)
    )

    # Gesture label
    if thumb_up:
        gesture = "THUMBS UP"
    elif finger_count == 0:
        gesture = "FIST"
    elif finger_count == 5:
        gesture = "OPEN PALM"
    elif finger_count == 1:
        gesture = "ONE FINGER"
    elif finger_count == 2:
        gesture = "TWO FINGERS"
    elif finger_count == 3:
        gesture = "THREE FINGERS"
    elif finger_count == 4:
        gesture = "FOUR FINGERS"
    else:
        gesture = "UNKNOWN"

    return finger_count, gesture, thumb_up


# -------------------------------
# MAIN PROGRAM
# -------------------------------
def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Webcam not available!")
        return

    total_faces = 0
    total_hands = 0
    total_fingers = 0
    thumbs_up_count = 0

    previous_time = time.time()

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        ) as hands:

            print("Face and Hand Counting Started")
            print("Press Q to exit")

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame")
                    break

                # Mirror webcam view
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                # -----------------------
                # FACE DETECTION
                # -----------------------
                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(60, 60)
                )

                total_faces = len(faces)

                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Face",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                # -----------------------
                # HAND DETECTION
                # -----------------------
                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb)

                total_hands = 0
                total_fingers = 0
                thumbs_up_count = 0

                if results.multi_hand_landmarks:
                    total_hands = len(
                        results.multi_hand_landmarks
                    )

                    for i, hand_landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                        handedness = "Right"

                        if results.multi_handedness:
                            handedness = (
                                results.multi_handedness[i]
                                .classification[0].label
                            )

                        fingers, gesture, thumb_up = (
                            analyze_hand(
                                hand_landmarks, handedness
                            )
                        )

                        total_fingers += fingers

                        if thumb_up:
                            thumbs_up_count += 1

                        # Position label above the wrist
                        wrist = hand_landmarks.landmark[0]
                        x = int(wrist.x * width)
                        y = max(
                            int(wrist.y * height) - 25, 25
                        )

                        cv2.putText(
                            frame,
                            f"{handedness}: {gesture}",
                            (max(x - 80, 5), y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Fingers: {fingers}",
                            (max(x - 80, 5), y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )

                # -----------------------
                # DISPLAY LIVE COUNTS
                # -----------------------
                cv2.rectangle(
                    frame, (0, 0), (310, 175),
                    (35, 35, 35), -1
                )

                cv2.putText(
                    frame, f"Faces: {total_faces}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

                cv2.putText(
                    frame, f"Hands: {total_hands}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 150, 0), 2
                )

                cv2.putText(
                    frame, f"Total Fingers: {total_fingers}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )

                cv2.putText(
                    frame, f"Thumbs Up: {thumbs_up_count}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 100, 255), 2
                )

                # FPS
                current_time = time.time()
                elapsed = current_time - previous_time
                fps = 1 / elapsed if elapsed > 0 else 0
                previous_time = current_time

                cv2.putText(
                    frame, f"FPS: {int(fps)}",
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )

                cv2.imshow(
                    "Face, Hand & Finger Counting",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Project stopped.")


if __name__ == "__main__":
    main()
