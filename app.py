from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import io
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Global variables to store image states
original_image = None
filtered_image = None
history = []
redo_stack = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_image_to_response(image):
    if image is None:
        return jsonify({'error': 'No image available'}), 400
    
    _, buffer = cv2.imencode('.png', image)
    response = io.BytesIO(buffer)
    response.seek(0)
    return send_file(response, mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/load_image', methods=['POST'])
def load_image():
    global original_image, filtered_image, history, redo_stack
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        original_image = cv2.imread(filepath)
        if original_image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Resize if needed
        max_width, max_height = 600, 350
        height, width = original_image.shape[:2]
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_size = (int(width * scale), int(height * scale))
            original_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_AREA)
        
        filtered_image = original_image.copy()
        history = [original_image.copy()]
        redo_stack = []
        
        # Return the URL to access the uploaded file
        return jsonify({
            'message': 'Image loaded successfully',
            'image_url': f'/uploads/{filename}'
        }), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/save_image', methods=['GET'])
def save_image():
    global filtered_image
    return save_image_to_response(filtered_image)

@app.route('/reset_image', methods=['GET'])
def reset_image():
    global original_image, filtered_image, history, redo_stack
    
    if original_image is None:
        return jsonify({'error': 'No original image available'}), 400
    
    filtered_image = original_image.copy()
    history = [original_image.copy()]
    redo_stack = []
    
    return save_image_to_response(filtered_image)

@app.route('/grayscale', methods=['GET'])
def grayscale():
    global original_image, filtered_image, history, redo_stack

    if original_image is None or filtered_image is None:
        return jsonify({'error': 'No image available'}), 400

    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    history.append(filtered_image.copy())
    redo_stack = []

    return save_image_to_response(filtered_image)

@app.route('/biner', methods=['GET'])
def biner():
    global original_image, filtered_image, history, redo_stack

    if original_image is None or filtered_image is None:
        return jsonify({'error': 'No image available'}), 400

    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    filtered_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    history.append(filtered_image.copy())
    redo_stack = []

    return save_image_to_response(filtered_image)

@app.route('/gaussian_filter', methods=['GET'])
def gaussian_filter():
    global original_image, filtered_image, history, redo_stack

    if original_image is None or filtered_image is None:
        return jsonify({'error': 'No image available'}), 400

    kernel_size = 3
    sigma = 1.4
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = (1/(2*np.pi*sigma**2)) * np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel = kernel / np.sum(kernel)
    filtered_image = cv2.filter2D(filtered_image, -1, kernel)
    history.append(filtered_image.copy())
    redo_stack = []

    return save_image_to_response(filtered_image)

@app.route('/histogram_equalization', methods=['GET'])
def histogram_equalization():
    global original_image,  filtered_image, history, redo_stack

    if original_image is None or filtered_image is None:
        return jsonify({'error': 'No image available'}), 400

    hist, bins = np.histogram(filtered_image.flatten(), 256, [0, 256]) 
    cdf = hist.cumsum() 
    cdf_normalized = cdf * hist.max() / cdf.max() 
    cdf_m = np.ma.masked_equal(cdf, 0) 
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) 
    cdf = np.ma.filled(cdf_m, 0).astype('uint8') 
    filtered_image = cdf[filtered_image] 
    history.append(filtered_image.copy())
    redo_stack = []

    return save_image_to_response(filtered_image)

@app.route('/show_histogram', methods=['GET'])
def show_histogram():
    global original_image, filtered_image

    if filtered_image is None or original_image is None:
        return jsonify({'error': 'No image available'}), 400

    plt.figure()
    hist, bins = np.histogram(filtered_image.flatten(), 256, [0, 256])
    plt.hist(filtered_image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')

@app.route('/undo', methods=['GET'])
def undo():
    global filtered_image, history, redo_stack
    
    if len(history) <= 1:
        return jsonify({'error': 'Nothing to undo'}), 400
    
    redo_stack.append(history.pop())
    filtered_image = history[-1].copy()
    
    return save_image_to_response(filtered_image)

@app.route('/redo', methods=['GET'])
def redo():
    global filtered_image, history, redo_stack
    
    if not redo_stack:
        return jsonify({'error': 'Nothing to redo'}), 400
    
    history.append(redo_stack.pop())
    filtered_image = history[-1].copy()
    
    return save_image_to_response(filtered_image)

@app.route('/adjust_image', methods=['POST'])
def adjust_image():
    global filtered_image
    
    if filtered_image is None:
        return jsonify({'error': 'No image available'}), 400
    
    data = request.json
    brightness = data.get('brightness', 0)
    contrast = data.get('contrast', 100) / 100.0
    sharpening = data.get('sharpening', 0)
    saturation = data.get('saturation', 100) / 100.0
    hue_shift = data.get('hue_shift', 0)
    value_scale = data.get('value_scale', 100) / 100.0

    image = filtered_image.copy()
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    if sharpening > 0:
        # Normalize sharpening value: 0 (no sharpen) to 100 (very sharp)
        # Kernel center: 9 + (sharpening / 10), so 0 -> 9, 100 -> 19
        kernel_strength = 9 + (sharpening / 10.0)
        kernel = np.array([[-1, -1, -1], 
                           [-1, kernel_strength, -1], 
                           [-1, -1, -1]], dtype=np.float32)
        image = cv2.filter2D(image, -1, kernel)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 2] *= value_scale
    hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return save_image_to_response(image)

@app.route('/save_original_pixels_excel', methods=['GET'])
def save_original_pixels_excel():
    global original_image
    
    if original_image is None:
        return jsonify({'error': 'No original image available'}), 400
    
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    df_r = pd.DataFrame(rgb_image[:,:,0])
    df_g = pd.DataFrame(rgb_image[:,:,1])
    df_b = pd.DataFrame(rgb_image[:,:,2])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_r.to_excel(writer, sheet_name='Red Channel')
        df_g.to_excel(writer, sheet_name='Green Channel')
        df_b.to_excel(writer, sheet_name='Blue Channel')
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='original_pixels.xlsx'
    )

@app.route('/save_edited_pixels_excel', methods=['GET'])
def save_edited_pixels_excel():
    global filtered_image
    
    if filtered_image is None:
        return jsonify({'error': 'No edited image available'}), 400
    
    rgb_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    df_r = pd.DataFrame(rgb_image[:,:,0])
    df_g = pd.DataFrame(rgb_image[:,:,1])
    df_b = pd.DataFrame(rgb_image[:,:,2])
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_r.to_excel(writer, sheet_name='Red Channel')
        df_g.to_excel(writer, sheet_name='Green Channel')
        df_b.to_excel(writer, sheet_name='Blue Channel')
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='edited_pixels.xlsx'
    )

@app.route('/save_original_pixels_text', methods=['GET'])
def save_original_pixels_text():
    global original_image
    
    if original_image is None:
        return jsonify({'error': 'No original image available'}), 400
    
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    
    output = io.StringIO()
    output.write("Original Image Pixel Values (RGB):\n")
    for i in range(height):
        for j in range(width):
            pixel = rgb_image[i,j]
            output.write(f"Pixel [{i},{j}]: R={pixel[0]}, G={pixel[1]}, B={pixel[2]}\n")
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    output.close()
    
    return send_file(
        mem,
        mimetype='text/plain',
        as_attachment=True,
        download_name='original_pixels.txt'
    )

@app.route('/save_edited_pixels_text', methods=['GET'])
def save_edited_pixels_text():
    global filtered_image
    
    if filtered_image is None:
        return jsonify({'error': 'No edited image available'}), 400
    
    rgb_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    
    output = io.StringIO()
    output.write("Edited Image Pixel Values (RGB):\n")
    for i in range(height):
        for j in range(width):
            pixel = rgb_image[i,j]
            output.write(f"Pixel [{i},{j}]: R={pixel[0]}, G={pixel[1]}, B={pixel[2]}\n")
    
    mem = io.BytesIO()
    mem.write(output.getvalue().encode('utf-8'))
    mem.seek(0)
    output.close()
    
    return send_file(
        mem,
        mimetype='text/plain',
        as_attachment=True,
        download_name='edited_pixels.txt'
    )

@app.route('/detect_face', methods=['GET'])
def detect_face():
    global filtered_image
    if filtered_image is None:
        return jsonify({'error': 'No image available'}), 400

    # Load haarcascade for face
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    img = filtered_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return save_image_to_response(img)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=5000, debug=True)