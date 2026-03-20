import os
import cv2
import concurrent.futures
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.face_model_arch import ResEmoteNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]


# ---------------- LOAD MODEL ----------------
def load_face_model(model_path):
    model = ResEmoteNet(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------- TRANSFORM ----------------
inference_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ---------------- EMOTION VECTOR ----------------
def get_emotion_vector(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    # Original model's actual training labels: 
    # {0: "Happy", 1: "Surprise", 2: "Sad", 3: "Anger", 4: "Disgust", 5: "Fear", 6: "Neutral"}
    # Target fusion array order: ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    reordered_probs = probs[[3, 4, 5, 0, 6, 2, 1]]
    return reordered_probs


# ---------------- MAIN FUNCTION ----------------
def get_face_emotion_from_image(image_path, model):

    try:
        img = cv2.imread(image_path)

        if img is None:
            print("Invalid image path")
            return np.zeros(7)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print("No face detected")
            return np.zeros(7)

        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        image = inference_transform(face).to(device)
        emotion_vector = get_emotion_vector(model, image)

        return emotion_vector

    except Exception as e:
        print(f"Face inference error: {e}")
        return np.zeros(7)

# ---------------- VIDEO PARALLEL FUNCTION ----------------
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
        
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face)
    tensor = inference_transform(face)
    return tensor

def get_face_emotion_from_video_v1(video_path, model, sample_rate_fps=5):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Invalid video path")
            return np.zeros(7)
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 # fallback
        
        frame_interval = max(int(fps / sample_rate_fps), 1)
        frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frames.append(frame)
            frame_count += 1
            
        cap.release()
        
        if not frames:
            return np.zeros(7)
            
        # Parallel face detection and transformation
        # with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        #     tensors = list(executor.map(process_frame, frames))
        
        # Sequential face detection and transformation to avoid cv2.CascadeClassifier thread-safety issues
        tensors = [process_frame(f) for f in frames]

        valid_tensors = [t for t in tensors if t is not None]
        
        if not valid_tensors:
            print("No face detected in video")
            return np.zeros(7)
            
        # Batched inference with batch_size=1 to prevent CUDA out of memory on constrained GPUs
        batch_size = 1
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(valid_tensors), batch_size):
                batch = valid_tensors[i:i + batch_size]
                batch_tensor = torch.stack(batch).to(device)
                outputs = model(batch_tensor)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(batch_probs)
        probs = np.array(all_probs)
            
        avg_probs = np.mean(probs, axis=0)
        
        reordered_probs = avg_probs[[3, 4, 5, 0, 6, 2, 1]]
        return reordered_probs
        
    except Exception as e:
        print(f"Face video inference error: {e}")
        return np.zeros(7)
