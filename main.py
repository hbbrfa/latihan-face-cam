import time
from datetime import datetime
from pathlib import Path

import cv2


WINDOW_NAME = "FACE CAM | Live Detection"
CAPTURE_DIR = Path(__file__).parent / "captures"
CYAN = (255, 220, 70)
WHITE = (240, 244, 248)
MUTED = (150, 165, 180)
PANEL = (22, 29, 39)


def draw_face_box(frame, x, y, width, height, index):
    color = CYAN
    corner = max(12, min(width, height) // 5)
    thickness = 2

    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)

    # Corner brackets make the detected face easier to spot in the preview.
    for start, end in (
        ((x, y), (x + corner, y)),
        ((x, y), (x, y + corner)),
        ((x + width, y), (x + width - corner, y)),
        ((x + width, y), (x + width, y + corner)),
        ((x, y + height), (x + corner, y + height)),
        ((x, y + height), (x, y + height - corner)),
        ((x + width, y + height), (x + width - corner, y + height)),
        ((x + width, y + height), (x + width, y + height - corner)),
    ):
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)

    label_y = max(22, y - 8)
    cv2.putText(
        frame,
        f"FACE {index:02d}",
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame, face_count, fps, elapsed, mirrored):
    height, width = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (width, 82), PANEL, -1)
    cv2.rectangle(overlay, (0, height - 46), (width, height), PANEL, -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    cv2.circle(frame, (25, 27), 6, (80, 235, 130), -1, cv2.LINE_AA)
    cv2.putText(
        frame, "LIVE", (40, 32), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (100, 245, 150), 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, "FACE CAM", (22, 62), cv2.FONT_HERSHEY_SIMPLEX,
        0.82, WHITE, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, "REAL-TIME FACE DETECTION", (width // 2 - 120, 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, MUTED, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"{face_count:02d} FACE{'S' if face_count != 1 else ''}",
        (width - 205, 31), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, CYAN, 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"{fps:4.0f} FPS  |  {elapsed}",
        (width - 205, 59), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, WHITE, 1, cv2.LINE_AA
    )

    mirror_status = "ON" if mirrored else "OFF"
    cv2.putText(
        frame,
        f"[Q] QUIT     [S] SCREENSHOT     [M] MIRROR: {mirror_status}",
        (18, height - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        datetime.now().strftime("%H:%M:%S"),
        (width - 100, height - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        MUTED,
        1,
        cv2.LINE_AA,
    )


def main():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Model deteksi wajah OpenCV tidak dapat dimuat.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Kamera tidak dapat dibuka. Periksa koneksi atau izin kamera.")

    mirrored = True
    start_time = time.perf_counter()
    previous_frame_time = start_time
    fps = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Gagal mengambil gambar dari kamera.")

            if mirrored:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )

            for index, (x, y, width, height) in enumerate(faces, start=1):
                draw_face_box(frame, x, y, width, height, index)

            now = time.perf_counter()
            frame_duration = now - previous_frame_time
            if frame_duration > 0:
                instant_fps = 1.0 / frame_duration
                fps = instant_fps if fps == 0 else fps * 0.85 + instant_fps * 0.15
            previous_frame_time = now

            elapsed_seconds = int(now - start_time)
            elapsed = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
            draw_hud(frame, len(faces), fps, elapsed, mirrored)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirrored = not mirrored
            elif key == ord("s"):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                filename = datetime.now().strftime("facecam_%Y%m%d_%H%M%S.png")
                screenshot_path = CAPTURE_DIR / filename
                if cv2.imwrite(str(screenshot_path), frame):
                    print(f"Screenshot tersimpan: {screenshot_path}")
                else:
                    print(f"Gagal menyimpan screenshot: {screenshot_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
