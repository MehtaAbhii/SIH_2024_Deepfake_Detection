from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import pipeline
import numpy as np
import soundfile as sf
from flask_cors import CORS
import cv2
import os
import tempfile
from werkzeug.utils import secure_filename
import json
import random


# Your ML model and functions
from transformers import AutoModelForImageClassification
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from groq import Groq

app = Flask(__name__)
CORS(app)

# Load the audio classification model with PyTorch
model = pipeline("audio-classification", model="Heem2/Deepfake-audio-detection", framework="pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'filea' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files['filea']
    data, samplerate = sf.read(audio_file)
    if len(data.shape) > 1:  # if stereo, take only one channel
        data = data[:, 0]
    data = np.array(data)
    result = model(data)
    print(result)
    return jsonify(result)



groq_api_key = "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"
client = Groq(api_key=groq_api_key)

model_name = "dima806/deepfake_vs_real_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

app = Flask(__name__)

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = preprocess(image)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)

def pixel_level_analysis(image_tensor):
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()


    texture_variance = np.var(image_np)


    noise_std = np.std(image_np)


    edges = np.mean(np.abs(np.gradient(image_np)))

    return {
        "Texture Variance": texture_variance,
        "Noise STD": noise_std,
        "Edge Sharpness": edges
    }


def facial_feature_analysis(image_tensor):


    center_region = image_tensor[:, :, 112-56:112+56, 112-56:112+56].squeeze().permute(1, 2, 0).numpy()

    facial_symmetry = np.var(np.flip(center_region, axis=1) - center_region)
    eye_lip_detail = np.mean(np.abs(np.gradient(center_region)))

    return {
        "Facial Symmetry": facial_symmetry,
        "Eye and Lip Detail": eye_lip_detail
    }


def lighting_and_shadows_analysis(image_tensor):
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()


    brightness = np.mean(image_np, axis=2)
    lighting_variance = np.var(brightness)


    shadow_gradient = np.mean(np.gradient(brightness))

    return {
        "Lighting Variance": lighting_variance,
        "Shadow Softness": shadow_gradient
    }


def color_analysis(image_tensor):
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()


    color_variance = np.var(image_np, axis=(0, 1))


    skin_tone_variance = np.var(image_np[90:130, 90:130, :])

    return {
        "Color Variance": color_variance.tolist(),
        "Skin Tone Uniformity": skin_tone_variance
    }


def geometric_consistency_analysis(image_tensor):
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()


    face_center = image_np[90:130, 90:130, :]

    proportions = np.mean(np.abs(np.gradient(face_center)))

    return {
        "Proportions Consistency": proportions,
    }


def artifact_detection(image_tensor):
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()


    checkerboard = np.mean(np.abs(np.gradient(np.gradient(image_np))))


    blurredness = np.std(image_np - np.clip(image_np, 0, 1))

    return {
        "Checkerboard Artifacts": checkerboard,
        "Blurring/Ghosting": blurredness
    }


def gan_specific_artifacts_analysis(image_tensor):
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()


    repeating_patterns = np.mean(np.abs(np.diff(image_np, axis=0))) + np.mean(np.abs(np.diff(image_np, axis=1)))

    return {
        "Repeating Patterns": repeating_patterns
    }


def latent_space_indicators_analysis(image_tensor):
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()


    variation = np.var(image_np)

    return {
        "Latent Space Variation": variation
    }

def extract_features(image_path):
    input_tensor = preprocess_image(image_path)

    pixel_analysis = pixel_level_analysis(input_tensor)
    facial_analysis = facial_feature_analysis(input_tensor)
    lighting_shadows = lighting_and_shadows_analysis(input_tensor)
    color_analysis_res = color_analysis(input_tensor)
    geometric_consistency = geometric_consistency_analysis(input_tensor)
    artifacts = artifact_detection(input_tensor)
    gan_artifacts = gan_specific_artifacts_analysis(input_tensor)
    latent_space = latent_space_indicators_analysis(input_tensor)

    combined_features = {
        **pixel_analysis,
        **facial_analysis,
        **lighting_shadows,
        **color_analysis_res,
        **geometric_consistency,
        **artifacts,
        **gan_artifacts,
        **latent_space
    }

    with torch.no_grad():
        outputs = model(input_tensor)

    return combined_features, outputs


def generate_report(features, classification_output):
    predicted_label = torch.argmax(classification_output.logits, dim=1).item()
    label_names = ["Real", "Fake"]
    predicted_label_name = label_names[predicted_label]

    detailed_explanation = ". ".join([f"{key}: {value}" for key, value in features.items()])

    prompt = (
        f"The image has been classified as '{predicted_label_name}' based on the extracted features. "
        f"The following analysis was performed: {detailed_explanation}. "
        f"Based on these features, analyze and explain the specific mathematical techniques likely used "
        f"in creating this deepfake, incorporating relevant deep learning methods, GAN architectures, "
        f"and any other pertinent details."
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gemma2-9b-it",   # Adjust model name if necessary
        max_tokens=5000
    )

    report = chat_completion.choices[0].message.content
    return report

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video-predict', methods=['POST'])
def video_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        video_path = os.path.join(upload_dir, file.filename)
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        frame_count = 0
        frames = []

        while success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            frame_path = os.path.join(upload_dir, f"frame_{frame_count}.jpg")
            pil_image.save(frame_path)
            frames.append(f"frame_{frame_count}.jpg")
            frame_count += 1
            success, image = cap.read()

        cap.release()

        if not frames:
            return jsonify({'error': 'No frames extracted from the video'}), 400

        random_frame_path = random.choice(frames)
        input_tensor = preprocess_image(os.path.join(upload_dir, random_frame_path))
        features, classification_output = extract_features(os.path.join(upload_dir, random_frame_path))

        features = {k: float(v) if isinstance(v, np.float32) else v for k, v in features.items()}
        report = generate_report(features, classification_output)

        result = {
            'selected_frame': random_frame_path,
            'report': report
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




if __name__ == '__main__':
    app.run(debug=True)
