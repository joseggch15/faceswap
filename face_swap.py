import os
# Corrección crítica: Fuerza a Windows a encontrar las librerías CUDA (ajusta la ruta si es necesario)
# Si instalaste CUDA en otra ruta, asegúrate de que esta carpeta contenga cublasLt64_12.dll
cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\bin"
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)

import cv2
import insightface
from insightface.app import FaceAnalysis

def main():
    print("Cargando modelos de Inteligencia Artificial en GPU... (Esto puede tomar unos segundos)")
    
    # 1. Inicializar el analizador de rostros con soporte para GPU
    # 'providers' obliga a buscar primero el soporte CUDA (NVIDIA)
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 2. Cargar el modelo generador de Face Swap con soporte para GPU
    try:
        swapper = insightface.model_zoo.get_model(
            'inswapper_128.onnx', 
            download=False, 
            download_zip=False, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    except Exception as e:
        print(f"❌ Error crítico al cargar el modelo 'inswapper_128.onnx': {e}")
        return

    # 3. Cargar la imagen de la Skin
    img_skin = cv2.imread('skin.jpg')
    if img_skin is None:
        print("❌ Error: No se encontró 'skin.jpg' en la carpeta.")
        return

    # Analizar la skin para extraer características biométricas
    skin_faces = app.get(img_skin)
    if not skin_faces:
        print("❌ Error: La IA no detectó ningún rostro claro en 'skin.jpg'.")
        return
    skin_face_data = skin_faces[0] 

    # 4. Iniciar la cámara
    cap = cv2.VideoCapture(0)
    print("🎥 Cámara iniciada. Usando aceleración por GPU. Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # La IA analiza tu rostro en vivo
        faces = app.get(frame)

        if faces:
            # Seleccionar el rostro detectado en la cámara
            target_face = faces[0]
            
            # MAGIA: Reemplazar el rostro detectado con los datos de la skin
            frame = swapper.get(frame, target_face, skin_face_data, paste_back=True)

        cv2.imshow('Face Swap Hiperrealista (IA - GPU)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()