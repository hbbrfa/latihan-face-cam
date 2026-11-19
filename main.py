
import cv2

# 1. Memuat model pendeteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# 2. Menyalakan kamera laptop
kamera = cv2.VideoCapture(0)

if not kamera.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

while True:
    # 3. Mengambil gambar dari kamera
    berhasil, frame = kamera.read()

    if not berhasil:
        print("Gagal mengambil gambar!")
        break

    # 4. Mengubah gambar menjadi grayscale
    abu_abu = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    # 5. Mendeteksi wajah
    wajah = face_cascade.detectMultiScale(
        abu_abu,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 6. Menggambar kotak di sekitar wajah
    for (x, y, w, h) in wajah:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # 7. Menampilkan jumlah wajah
    cv2.putText(
        frame,
        f"Jumlah wajah: {len(wajah)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # 8. Menampilkan kamera
    cv2.imshow("Face Detection - AI Project", frame)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 9. Membersihkan kamera
kamera.release()
cv2.destroyAllWindows()
