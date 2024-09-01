# SIH_2024_Deepfake_Detection

---





https://github.com/user-attachments/assets/a24d3657-57b1-4f81-bdcb-5fd81b2c7a7b

## Problem Statement ID

```
Team : O(log n)
```
```
Problem Statement ID : 1683
```

---

## Problem Statement

```
Development of AI/ML based solution for detection of face-swap based deep fake videos
```

---

# DeFaceGuard

```
A comprehensive solution with accessible and user-friendly Android app as well as Web interface to detect and analyze deepfake videos
```

---

## Features

```
1. Detailed analysis of the video to detect deepfake content with attention map of facial anomalies
2. Detailed report generation , justifying the model's decision
3. Multi-modal approach to detect deepfake videos using both audio and visual cues
4. User-friendly Android app for easy access to the solution instant and on-the-go
5. Web interface for on-device analysis of deepfake videos, protecting user privacy
```

---

## Technology Stack

```
We have used the following technologies to build our solution:

1. CNN + LSTM approach for deepfake detection :
    The CNN model is used to extract features from the frames of the video and LSTM is used to analyze the temporal features of the video.
2. Audio analysis:
    This feature is currently under development and will be integrated soon.
    For now , we are using a pretrained model to determine whether the audio is real or fake.
    We expect to extract features from the audio as well and pass it to LLM for further analysis.

3. Android App:
    The Android app is being built using flutter and dart.
    We have used Hugging Face's transformers library to integrate the model in the app.
    The app will be able to analyze the video and provide the user with the results.
    The app will also have a feature to analyze the audio of the video.
    
4. Web Interface:
    The web interface is being built using Django.
    The user can upload the video and the model will analyze the video and provide the user with the results.
    Attention map of the video is ready with the model, but its integration with the web interface is under progress.
    Audio analysis provides accurate results and is being integrated with the web interface.
```

---

## Installation

#### 1. Clone the repository
```bash
git clone https://github.com/MehtaAbhii/SIH_2024_Deepfake_Detection.git
```

#### 2. Select the interface and change directory accordingly
```bash
cd Web
```

```bash
cd App
```

#### 3. Install the required dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the application

a. For Web interface
```bash
python manage.py runserver
```

b. For Android app
```bash
flutter run
```

#### 5. Follow the on-screen instructions to use the application

---

## Team Members

1. Abhi Mehta
2. Aditya Yedurkar
3. Anushka Yadav
4. Akshita Bhasin
5. Krish Porwal
6. Sanhita Rajput

---
## Technologies Used

#### Web Development
Languages : HTML, CSS, JavaScript, Python
Frameworks : Django
Database: PostgreSQL, SQLite
#### App Development
Flask
Flutter
Dart

#### AI/ML
Deep Learning Models : ResNeXt CNN, LSTM, LLM
Visualization : Grad-CAM
Advanced NLP: Transformers 

-----

## Future Scope

1. Integration of audio analysis with the model
2. Attention map integration with the web interface
3. Real-time analysis of the video
4. Integration of already implemented features with the app and web interface
    These features include:
    - Voice delay detection
    - Video transcript anaysis
    - lipsync detection
5. Deployment of the solution on cloud for easy access and scalability











