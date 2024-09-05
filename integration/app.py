from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification
from torchvision import transforms
from collections import Counter
from werkzeug.utils import secure_filename
from groq import Groq
from model import VideoGanAnalyzer  # Assuming you saved your provided GAN model code in a separate file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/analyzed_videos'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'gif'}

groq_api_key = "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"
client = Groq(api_key=groq_api_key)

model_name = "dima806/deepfake_vs_real_image_detection"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = preprocess(image)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)

def extract_features(frame):
    input_tensor = preprocess_frame(frame)
    with torch.no_grad():
        outputs = model(input_tensor)
    return input_tensor.squeeze(), outputs

def map_features_to_concepts(features):
    concept_summary = {}
    mean_feature_value = features.mean().item()
    std_feature_value = features.std().item()

    concept_summary['Skin Texture'] = mean_feature_value
    concept_summary['Texture Variance'] = std_feature_value
    concept_summary['Lighting'] = mean_feature_value
    concept_summary['Facial Expression'] = abs(mean_feature_value)
    concept_summary['Edge Characteristics'] = std_feature_value

    return concept_summary

def process_video(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    processed_frames = 0

    all_features = []
    classifications = []
    concept_summaries = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            features, classification_output = extract_features(frame)
            all_features.append(features.cpu().numpy())

            predicted_label = torch.argmax(classification_output.logits, dim=1).item()
            classifications.append(predicted_label)

            concept_summaries.append(map_features_to_concepts(features))

            processed_frames += 1

        frame_count += 1

    cap.release()
    return np.array(all_features), classifications, concept_summaries, fps, total_frames

def analyze_video(video_path, frame_interval=30):
    all_features, classifications, concept_summaries, fps, total_frames = process_video(video_path, frame_interval)

    mean_features = np.mean(all_features, axis=0)
    std_features = np.std(all_features, axis=0)

    classification_counts = Counter(classifications)
    total_processed = len(classifications)
    fake_percentage = (classification_counts[1] / total_processed) * 100
    real_percentage = (classification_counts[0] / total_processed) * 100

    avg_concept_summary = {
        key: np.mean([summary[key] for summary in concept_summaries])
        for key in concept_summaries[0]
    }

    summary = f"""
    Video Analysis Summary:
    - Total frames: {total_frames}
    - Frames processed: {total_processed}
    - Video duration: {total_frames/fps:.2f} seconds
    - Fake frames detected: {fake_percentage:.2f}%
    - Real frames detected: {real_percentage:.2f}%
    - Mean feature value: {np.mean(mean_features):.4f}
    - Mean feature std dev: {np.mean(std_features):.4f}
    - Max feature value: {np.max(mean_features):.4f}
    - Min feature value: {np.min(mean_features):.4f}

    Average Concept Analysis:
    - Skin Texture: {avg_concept_summary['Skin Texture']:.4f}
    - Texture Variance: {avg_concept_summary['Texture Variance']:.4f}
    - Lighting: {avg_concept_summary['Lighting']:.4f}
    - Facial Expression: {avg_concept_summary['Facial Expression']:.4f}
    - Edge Characteristics: {avg_concept_summary['Edge Characteristics']:.4f}
    """

    prompt = f"""
    Based on the following deepfake detection analysis, provide a comprehensive report with the best possible conclusions:

    {summary}

    Your task is to:
    Directly provide answers without rewriting prompt language in output and provide bullet points, keep the answers precise.
    1. Evaluate the overall likelihood of the video being a deepfake, taking into account the statistical analysis and concept summaries.
    2. Interpret Skin Texture, Texture Variance, Lighting, Facial Expression, and Edge Characteristics in terms of what they reveal about the video's authenticity in detail in bullet points one after other.
    3. Identify deepfake techniques (mention their name, do not generalize) that are responsible for anomalies in the video, based on the feature and concept analysis.
    4. Summarize the findings in a clear, conclusive manner, ensuring that the analysis is thorough and insightful.
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="mixtral-8x7b-32768",
        max_tokens=5000
    )

    report = chat_completion.choices[0].message.content
    return report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
        
        # Perform GAN analysis
        model_path = r"models/MesoInception_DF.h5"
        analyzer = VideoGanAnalyzer(filepath, model_path)
        analyzer.main()

        try:
            report = analyze_video(filepath)
        finally:
            os.remove(filepath)

        # Pass the report and analyzed video path to the template
        return render_template(
            'analysis_report.html',
            report=report,
            analyzed_video=f"{filename.rsplit('.', 1)[0]}_analyzed.avi"
        )
    return jsonify({"error": "Invalid file format"}), 400


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)