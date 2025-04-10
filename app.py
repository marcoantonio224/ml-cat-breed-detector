import os
import json
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from breed_detector_model import predict_breed, get_training_data, get_accuracy_and_confidence_list

app = Flask(__name__)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Preload accuracy vs confidence data
accuracies, confidences, average_accuracy = get_accuracy_and_confidence_list()


# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    training_data = get_training_data()
    return render_template(
        'index.html',
        training_data=json.dumps(training_data),
        line_chart_data=json.dumps({
            "accuracies": accuracies,
            "confidences": confidences,
            "average_accuracy": average_accuracy
        })
    )


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})   
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)     
        # Call the predict function to get the breed and confidence
        top_breeds, top_confidences = predict_breed(file_path)
        # Get predictions
        predictions = []
        for breed, confidence in list(zip(top_breeds, top_confidences)):
            confidence = round(confidence, 2)
            if confidence <= 0.0:
                continue
            predictions.append({"breed": breed, "confidence": str(confidence)})
        # Generate the URL for the uploaded image
        image_url = url_for('uploaded_file', filename=filename)
        return render_template(
            'prediction.html',
            image_url=image_url,
            predictions=json.dumps({"results": predictions}),
        )
    else:
        return jsonify({'error': 'Invalid file type. Only jpg, jpeg, and png are allowed.'})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('./uploads', filename)


if __name__ == '__main__':
    app.run(debug=True)