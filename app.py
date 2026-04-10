import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("models/deepfake_model.h5")

#upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    file = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    
    filename = file.filename.lower()
    is_video = False
    
    #-----------------IMAGE----------------
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        
        print("Prediction Score:", score)
        
        result = "FAKE" if score > 0.5 else "REAL"
        score = round(score * 100, 2)

    #---------------VIDEO-------------------
    elif filename.endswith((".mp4", ".avi", ".mov")):

        is_video = True
        cap = cv2.VideoCapture(file_path)

        frame_count = 0
        fake_count = 0
        total_checked = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 🔥 Check every 15th frame (reduce noise)
            if frame_count % 15 == 0:
                frame = cv2.resize(frame, (128, 128))

                # 🔥 IMPORTANT color fix
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = frame / 255.0
                frame = np.expand_dims(frame, axis=0)

                prediction = model.predict(frame)
                score_frame = float(prediction[0][0])

                # Same logic as image
                if score_frame > 0.5:
                    fake_count += 1

                total_checked += 1

            frame_count += 1

        cap.release()

        # Avoid division error
        fake_ratio = fake_count / total_checked if total_checked > 0 else 0

        print("Fake Ratio:", fake_ratio)

        # 🔥 IMPORTANT threshold fix
        if fake_ratio > 0.7:
            result = "FAKE VIDEO"
        else:
            result = "REAL VIDEO"

        score = round(fake_ratio * 100, 2)
    else:
        return "Unsupported File Format"
        
    return render_template("result.html", filename=file.filename, prediction=result, score=score, file_path=file_path, is_video=is_video)
    
if __name__ == "__main__":
    app.run(debug=True)