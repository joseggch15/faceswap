import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# FUNCIONES MATEMÁTICAS Y DE DEFORMACIÓN
# ==========================================

def get_landmarks(img, face_mesh):
    """Obtiene los puntos faciales de una imagen usando MediaPipe."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    
    landmarks = []
    h, w, _ = img.shape
    for point in results.multi_face_landmarks[0].landmark:
        x = int(point.x * w)
        y = int(point.y * h)
        landmarks.append((x, y))
    return np.array(landmarks, np.int32)

def apply_affine_transform(src, src_tri, dst_tri, size):
    """Calcula y aplica la deformación matemática de un triángulo a otro."""
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def warp_triangle(img1, img2, t1, t2):
    """Recorta, deforma y pega un triángulo de la imagen origen a la destino."""
    # 1. Encontrar los cuadros delimitadores (bounding boxes) de los triángulos
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # 2. Desplazar los puntos relativos a los bounding boxes
    t1_rect = []
    t2_rect = []
    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # 3. Recortar el triángulo de la imagen original (skin)
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    # 4. Deformar (Warp) el triángulo recortado para que coincida con la cámara
    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    # 5. Crear una máscara para el triángulo deformado
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)

    # 6. Pegar el triángulo deformado en la imagen final
    img2_rect = img2_rect * mask
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect

# ==========================================
# FLUJO PRINCIPAL
# ==========================================

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False)

    # PASO 1: CARGAR LA IMAGEN DE SKIN Y OBTENER PUNTOS
    img_skin = cv2.imread('skin.jpg')
    if img_skin is None:
        print("Error: No se encontró 'skin.jpg'.")
        return
    
    skin_landmarks = get_landmarks(img_skin, face_mesh)
    if skin_landmarks is None:
        print("Error: No se detectó rostro en la imagen de skin.")
        return

    # PASO 2: CREAR TRIÁNGULOS DELAUNAY PARA LA SKIN
    rect = (0, 0, img_skin.shape[1], img_skin.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in skin_landmarks:
        subdiv.insert((int(p[0]), int(p[1])))
    
    # Extraer índices de los triángulos
    triangles_list = subdiv.getTriangleList()
    delaunay_tri_indexes = []
    for t in triangles_list:
        pt = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        # Encontrar el índice original de cada punto del triángulo
        ind = []
        for j in range(3):
            for k in range(len(skin_landmarks)):
                if abs(pt[j][0] - skin_landmarks[k][0]) < 1.0 and abs(pt[j][1] - skin_landmarks[k][1]) < 1.0:
                    ind.append(k)
        if len(ind) == 3:
            delaunay_tri_indexes.append((ind[0], ind[1], ind[2]))

    # INICIAR CÁMARA
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_landmarks = get_landmarks(frame, face_mesh)

        # Si se detecta un rostro en la cámara
        if frame_landmarks is not None:
            # Crear una imagen vacía (lienzo) para construir el rostro deformado
            warped_face = np.zeros_like(frame)

            # PASO 3: DEFORMAR (WARP) CADA TRIÁNGULO
            for tri_indices in delaunay_tri_indexes:
                # Puntos del triángulo en la skin
                t_skin = [skin_landmarks[tri_indices[0]], skin_landmarks[tri_indices[1]], skin_landmarks[tri_indices[2]]]
                # Puntos del triángulo en la cámara
                t_frame = [frame_landmarks[tri_indices[0]], frame_landmarks[tri_indices[1]], frame_landmarks[tri_indices[2]]]
                
                warp_triangle(img_skin, warped_face, t_skin, t_frame)

            # PASO 4: CREAR UNA MÁSCARA DEL ROSTRO
            # Obtener el contorno exterior del rostro en la cámara (Convex Hull)
            hull = cv2.convexHull(frame_landmarks)
            mask = np.zeros_like(frame_gray)
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.merge([mask, mask, mask]) # Hacerla de 3 canales

            # PASO 5: MEZCLAR LA SKIN CON EL VIDEO ORIGINAL (Seamless Clone)
            # Encontrar el centro del rostro para la función de mezcla
            r = cv2.boundingRect(hull)
            center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
            
            try:
                # Mezclado suave de colores e iluminación
                output = cv2.seamlessClone(warped_face, frame, mask, center, cv2.NORMAL_CLONE)
            except cv2.error:
                # Si hay un error al mezclar (ej. el rostro sale de la pantalla), mostrar frame normal
                output = frame
        else:
            output = frame

        cv2.imshow('Face Swap en Tiempo Real', output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()