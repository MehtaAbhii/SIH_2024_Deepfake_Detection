from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from integration.model import VideoGanAnalyzer  # Assuming you saved your provided code in a separate file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/analyzed_videos'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform GAN analysis
            model_path = r"models\MesoInception_DF.h5"
            analyzer = VideoGanAnalyzer(filepath, model_path)
            analyzer.main()

            analyzed_filename = filename.rsplit('.', 1)[0] + "_analyzed.avi"
            return redirect(url_for('uploaded_file', filename=analyzed_filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
