from flask import Flask, request, jsonify, send_file, render_template
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)

DEVICE = 'cpu'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

# Ensure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize MTCNN and InceptionResnetV1 models
print("Initializing models...")
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()
print("Models initialized.")

def predict_single_frame(input_image: Image.Image):
    print("Predicting single frame...")
    face = mtcnn(input_image)
    if face is None:
        print("No face detected.")
        return None, None

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE).float() / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()

    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)[0, :]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"

        confidences = {
            'real': 1 - output.item(),
            'fake': output.item()
        }
    print(f"Prediction: {prediction}, Confidences: {confidences}")
    return confidences, face_with_mask

def resize_frame(image, size=(256, 256)):
    return cv2.resize(image, size)


def create_explainable_video(video_path, output_path, frame_skip=2, resize_size=(256, 256)):
    print(f"Processing video: {video_path}")
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_predictions = []
    
    for frame_num in range(0, total_frames, frame_skip):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = vidcap.read()
        
        if not success:
            print(f"Failed to read frame {frame_num}")
            continue
        
        # Resize the frame before processing
        resized_image = resize_frame(image, size=resize_size)
        pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        face = mtcnn(pil_image)
        
        if face is not None:
            face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
            face = face.astype('uint8')

            confidences, face_with_mask = predict_single_frame(pil_image)
            frame_predictions.append(confidences['fake'])
            
            # Ensure face_with_mask is resized to match face dimensions
            face_with_mask_resized = cv2.resize(face_with_mask, (face.shape[1], face.shape[0]))

            # Combine input and output side by side
            combined_frame = np.hstack((face, face_with_mask_resized))
            
            # Initialize video writer with the combined frame size
            if out is None:
                height, width, _ = combined_frame.shape
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                print(f"Video writer initialized: {output_path} with size ({width}, {height})")
            
            # Add text with prediction and confidence
            prediction = "Fake" if confidences['fake'] > 0.5 else "Real"
            confidence = confidences['fake'] if confidences['fake'] > 0.5 else confidences['real']
            text = f"{prediction}: {confidence:.2f}"
            cv2.putText(combined_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(combined_frame)
        else:
            print(f"No face detected in frame {frame_num}")

    vidcap.release()
    if out:
        out.release()
    print(f"Video processing completed. Output saved to {output_path}")
    
    if not frame_predictions:
        raise Exception('No faces detected in the video')
    
    avg_prediction = sum(frame_predictions) / len(frame_predictions)
    final_prediction = "fake" if avg_prediction > 0.5 else "real"
    confidences = {
        'real': 1 - avg_prediction,
        'fake': avg_prediction
    }
    
    return confidences, output_path



@app.route('/')
def index():
    print("Rendering upload page...")
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    print("Uploading video...")
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded file in the current directory
    file_path = os.path.join(UPLOAD_FOLDER, 'uploaded_video.mp4')
    file.save(file_path)
    print(f"Uploaded file saved to {file_path}")

    # Process the video
    try:
        output_path = os.path.join(PROCESSED_FOLDER, 'processed_video.mp4')
        
        confidences, output_video_path = create_explainable_video(file_path, output_path)
        print(f"Video processed. Output path: {output_video_path}")

        # Return results
        return render_template(
            'upload.html',
            prediction_class="fake" if confidences['fake'] > 0.5 else "real",
            confidence_real=confidences['real'],
            confidence_fake=confidences['fake'],
            video_url=f"/video/{os.path.basename(output_video_path)}"
        )

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/video/<filename>')
def serve_video(filename):
    video_path = os.path.join(PROCESSED_FOLDER, filename)
    print(f"Serving video: {video_path}")
    return send_file(video_path, mimetype='video/mp4')

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
