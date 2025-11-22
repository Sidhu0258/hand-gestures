# hand_finger_count_pygame.py
import time
from collections import deque

import numpy as np
import mediapipe as mp
import pygame
import pygame.camera

# --- Settings ---
CAMERA_INDEX = 0          # change if you have multiple cameras
WINDOW_TITLE = "Hand Detection + Finger Count (Pygame)"
SMOOTH_HISTORY = 5        # number of frames to average counts for smoothing
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

TIP_IDS = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks, img_w, img_h, handedness_label, flip_horizontal=True):
    """
    Count fingers for a single hand landmarks object.
    Returns: (count, finger_flags_list, wrist_pixel_xy)
    """
    lm = hand_landmarks.landmark
    coords = [(int(lm[i].x * img_w), int(lm[i].y * img_h)) for i in range(21)]

    fingers = []

    # Thumb detection using X positions, taking into account flip (selfie) or not
    thumb_tip_x = coords[4][0]
    thumb_ip_x  = coords[3][0]

    # Determine what direction means 'open' depending on handedness and flip
    if flip_horizontal:
        # flipped image: left/right appearances swap
        if handedness_label == "Right":
            thumb_open = 1 if thumb_tip_x > thumb_ip_x else 0
        else:
            thumb_open = 1 if thumb_tip_x < thumb_ip_x else 0
    else:
        if handedness_label == "Right":
            thumb_open = 1 if thumb_tip_x < thumb_ip_x else 0
        else:
            thumb_open = 1 if thumb_tip_x > thumb_ip_x else 0

    fingers.append(thumb_open)

    # Other fingers: tip y < pip y means finger is up (y grows downwards)
    for tip_id, pip_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if coords[tip_id][1] < coords[pip_id][1] else 0)

    # wrist pixel for placing text
    wrist = coords[0]

    return sum(fingers), fingers, wrist

def np_surface_from_pygame_surface(surf):
    """ Convert a pygame Surface to a HxWx3 uint8 numpy array in RGB order """
    arr = pygame.surfarray.array3d(surf)  # shape (w, h, 3)
    arr = np.transpose(arr, (1, 0, 2))    # to (h, w, 3)
    return arr

def pygame_surface_from_np(img_np):
    """ Convert HxWx3 uint8 numpy array (RGB) to pygame Surface """
    # pygame expects array shape (w,h,3) for make_surface
    arr = np.transpose(img_np, (1, 0, 2)).copy()
    return pygame.surfarray.make_surface(arr)

def main():
    pygame.init()
    pygame.camera.init()

    cams = pygame.camera.list_cameras()
    if not cams:
        raise RuntimeError("No camera found by pygame.camera. Make sure a webcam is connected.")
    cam_name = cams[CAMERA_INDEX % len(cams)]
    cam = pygame.camera.Camera(cam_name, (FRAME_WIDTH, FRAME_HEIGHT))
    cam.start()

    screen = pygame.display.set_mode((FRAME_WIDTH, FRAME_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    font = pygame.font.SysFont("Arial", 20)
    small_font = pygame.font.SysFont("Arial", 16)

    smoothing = deque(maxlen=SMOOTH_HISTORY)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        clock = pygame.time.Clock()
        running = True
        while running:
            start = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

            # get a frame from camera as pygame Surface
            frame_surf = cam.get_image()  # returns a Surface sized by camera init
            # Convert to numpy array (h, w, 3), RGB
            img_rgb = np_surface_from_pygame_surface(frame_surf)

            # Optional: flip horizontally for selfie view (makes handedness labeling intuitive)
            img_rgb = np.fliplr(img_rgb).copy()
            img_h, img_w = img_rgb.shape[:2]

            # MediaPipe expects RGB uint8
            results = hands.process(img_rgb)

            total = 0
            per_hand_texts = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, hand_handedness in zip(results.multi_hand_landmarks,
                                                     results.multi_handedness):
                    label = hand_handedness.classification[0].label  # "Left" / "Right"
                    cnt, flags, wrist = count_fingers(hand_lms, img_w, img_h, label, flip_horizontal=True)
                    total += cnt
                    per_hand_texts.append((label, cnt, wrist))

                    # Draw landmarks onto the numpy image using MediaPipe draw utils
                    # mp_draw modifies the image in-place
                    mp_draw.draw_landmarks(
                        img_rgb,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # smoothing
            smoothing.append(total)
            smooth_total = int(round(sum(smoothing) / len(smoothing)))

            # Convert numpy image back to pygame surface for display
            out_surf = pygame_surface_from_np(img_rgb)

            # blit to screen and overlay text
            screen.blit(out_surf, (0, 0))

            # Overlay per-hand counts
            for label, cnt, wrist in per_hand_texts:
                wx, wy = wrist
                # We flipped image earlier, so adjust x coordinate to pygame coords
                # wrist x is in pixels already (for flipped image), so use directly
                txt = f"{label}: {cnt}"
                text_surf = font.render(txt, True, (0, 255, 0))
                # clamp position to window
                tx = max(0, min(img_w - text_surf.get_width(), wx - 40))
                ty = max(0, min(img_h - text_surf.get_height(), wy - 40))
                screen.blit(text_surf, (tx, ty))

            # Overlay totals and FPS
            fps = clock.get_fps() if clock.get_fps() > 0 else 0.0
            info = f"Total Fingers (smoothed): {smooth_total}   FPS: {fps:.1f}   Press q or ESC to quit"
            info_surf = small_font.render(info, True, (255, 255, 255))
            screen.blit(info_surf, (8, 8))

            pygame.display.flip()
            clock.tick(30)  # limit to ~30 FPS

        cam.stop()
        pygame.quit()

if __name__ == "__main__":
    main()
