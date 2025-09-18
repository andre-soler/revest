import cv2
import mediapipe as mp
import platform
import time

def open_camera(preferred_index=None):
    """Tenta abrir a câmera com backends diferentes (Windows/Linux/macOS)."""
    sys = platform.system()
    if sys == "Windows":
        apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys == "Darwin":  # macOS
        apis = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:  # Linux
        apis = [cv2.CAP_V4L2, cv2.CAP_ANY]

    indices = list(range(0, 6))
    if preferred_index is not None and preferred_index in indices:
        indices.remove(preferred_index)
        indices = [preferred_index] + indices

    for api in apis:
        for idx in indices:
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap, f"idx={idx}, api={api}"
                cap.release()
    return None, "Nenhuma câmera disponível"

def main():
    # Haar Cascade direto do OpenCV (não precisa baixar XML)
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        raise RuntimeError(f"Não foi possível carregar o Haar em: {haar_path}")

    # Inicializa MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap, info = open_camera()
    if cap is None:
        raise RuntimeError("Não consegui abrir a câmera. Verifique permissões e se outro app está usando.")

    print(f"[INFO] Câmera aberta com {info}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    prev = time.time()
    print("Pressione 'q' para sair.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Haar (caixa verde) ----
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Haar", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---- MediaPipe (landmarks) ----
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lm,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=lm,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                )

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev) if (now - prev) > 0 else 0.0
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Haar + MediaPipe", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

