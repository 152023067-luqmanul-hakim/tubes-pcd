from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import pandas as pd
import io
from werkzeug.utils import secure_filename
import os
from math import hypot
import mediapipe as mp
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Update path untuk model
# HAAR_CASCADE_PATH = os.path.join(app.root_path, 'static', 'models', 'haarcascade_frontalface_default.xml')
# SHAPE_PREDICTOR_PATH = os.path.join(app.root_path, 'static', 'models', 'shape_predictor_68_face_landmarks.dat')
# Inisialisasi Face Landmarker
FACE_LANDMARKER_PATH = os.path.join(app.root_path, 'static', 'models', 'face_landmarker.task')

# Initialize face detection models
# haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
# shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
# dlib_detector = dlib.get_frontal_face_detector()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True
)

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

# def hybrid_face_detection(image):
#     """Gabungkan Haar Cascade dan Dlib detector untuk hasil terbaik"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Deteksi dengan Haar Cascade
#     haar_faces = haar_cascade.detectMultiScale(gray, 1.1, 5)
    
#     # Deteksi dengan Dlib
#     dlib_faces = dlib_detector(gray, 1)
    
#     # Gabungkan hasil deteksi
#     faces = []
    
#     # Konversi hasil Haar Cascade ke format dlib rectangle
#     for (x, y, w, h) in haar_faces:
#         faces.append(dlib.rectangle(int(x), int(y), int(x+w), int(y+h)))
    
#     # Tambahkan hasil deteksi dlib
#     for face in dlib_faces:
#         faces.append(face)
    
#     return faces

# def get_landmarks(image, face_rect):
#     """Mendapatkan landmark wajah dari ROI yang terdeteksi"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     landmarks = shape_predictor(gray, face_rect)
#     return [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

# def calculate_expression_metrics(landmarks):
#     """Menghitung parameter ekspresi dari landmark wajah"""
#     metrics = {}
    
#     # Fungsi bantu untuk eye aspect ratio
#     def eye_aspect_ratio(eye_points):
#         A = hypot(eye_points[1][0]-eye_points[5][0], eye_points[1][1]-eye_points[5][1])
#         B = hypot(eye_points[2][0]-eye_points[4][0], eye_points[2][1]-eye_points[4][1])
#         C = hypot(eye_points[0][0]-eye_points[3][0], eye_points[0][1]-eye_points[3][1])
#         return (A + B) / (2.0 * C)
    
#     # Mata kiri (points 36-41)
#     left_eye = [landmarks[36], landmarks[37], landmarks[38], 
#                 landmarks[39], landmarks[40], landmarks[41]]
#     metrics['left_eye'] = eye_aspect_ratio(left_eye)
    
#     # Mata kanan (points 42-47)
#     right_eye = [landmarks[42], landmarks[43], landmarks[44],
#                  landmarks[45], landmarks[46], landmarks[47]]
#     metrics['right_eye'] = eye_aspect_ratio(right_eye)
    
#     # Mulut (points 48-67)
#     mouth_width = hypot(landmarks[54][0]-landmarks[48][0], landmarks[54][1]-landmarks[48][1])
#     mouth_height = hypot(landmarks[57][0]-(landmarks[51][0]+landmarks[53][0])/2, 
#                         landmarks[57][1]-(landmarks[51][1]+landmarks[53][1])/2)
#     metrics['mouth_ratio'] = mouth_height / mouth_width
#     metrics['mouth_width'] = mouth_width
    
#     # Alis (points 17-26)
#     left_eyebrow = hypot(landmarks[21][0]-landmarks[17][0], landmarks[21][1]-landmarks[17][1])
#     right_eyebrow = hypot(landmarks[22][0]-landmarks[26][0], landmarks[22][1]-landmarks[26][1])
#     metrics['eyebrow_ratio'] = (left_eyebrow + right_eyebrow) / 2
    
#     # Lebar wajah sebagai referensi
#     metrics['face_width'] = hypot(landmarks[0][0]-landmarks[16][0], landmarks[0][1]-landmarks[16][1])
    
#     return metrics

# def determine_expression(metrics):
#     """Menentukan ekspresi berdasarkan parameter wajah"""
#     avg_eye = (metrics['left_eye'] + metrics['right_eye']) / 2
#     mouth_ratio = metrics['mouth_ratio']
#     mouth_width = metrics['mouth_width']
#     face_width = metrics['face_width']
#     eyebrow_ratio = metrics['eyebrow_ratio']
    
#     # Terkejut: mata dan mulut terbuka lebar
#     if avg_eye > 0.3 and mouth_ratio > 0.35:
#         return "Terkejut"
    
#     # Senang: mulut lebar dengan sudut bibir ke atas
#     elif mouth_ratio > 0.25 and mouth_width > 0.3 * face_width:
#         return "Senang"
    
#     # Marah: alis turun, mulut mengerucut
#     elif eyebrow_ratio < 0.8 and mouth_ratio < 0.15:
#         return "Marah"
    
#     # Sedih: alis bagian dalam naik, mata sedikit tertutup
#     elif eyebrow_ratio > 1.3 and avg_eye < 0.22:
#         return "Sedih"
    
#     # Takut: mata terbuka lebar, mulut tidak terlalu lebar
#     elif avg_eye > 0.28 and mouth_ratio < 0.2:
#         return "Takut"
    
#     # Jijik: mulut mengerucut vertikal
#     elif mouth_ratio > 0.3 and mouth_width < 0.25 * face_width:
#         return "Jijik"
    
#     else:
#         return "Netral"

# @app.route('/detect_expression', methods=['POST'])
# def detect_expression():
#     """Endpoint untuk deteksi ekspresi wajah"""
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         image = cv2.imread(filepath)
#         if image is None:
#             return jsonify({'error': 'Failed to read image'}), 400
        
#         # Resize if needed
#         max_width, max_height = 600, 350
#         height, width = image.shape[:2]
#         if width > max_width or height > max_height:
#             scale = min(max_width / width, max_height / height)
#             new_size = (int(width * scale), int(height * scale))
#             image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
#         # Deteksi wajah hybrid
#         faces = hybrid_face_detection(image)
        
#         if not faces:
#             return jsonify({'error': 'Failed to detect face or expression'}), 400
        
#         # Pilih wajah terbesar
#         main_face = max(faces, key=lambda rect: rect.width() * rect.height())
        
#         # Dapatkan landmarks
#         landmarks = get_landmarks(image, main_face)
        
#         # Analisis ekspresi
#         metrics = calculate_expression_metrics(landmarks)
#         expression = determine_expression(metrics)
        
#         # Gambar hasil deteksi
#         x1, y1, x2, y2 = main_face.left(), main_face.top(), main_face.right(), main_face.bottom()
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         for (x, y) in landmarks:
#             cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#         cv2.putText(image, expression, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
#         # Simpan gambar hasil deteksi
#         result_filename = f"result_{filename}"
#         result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
#         cv2.imwrite(result_path, image)
        
#         return jsonify({
#             'expression': expression,
#             'metrics': metrics,
#             'result_image_url': f'/uploads/{result_filename}'
#         }), 200
#     else:
#         return jsonify({'error': 'File type not allowed'}), 400

# Fungsi untuk memproses ekspresi wajah
def process_expression(face_blendshapes):
    # Inisialisasi skor ekspresi
    expression_scores = {
        # 'neutral': 0,
        'happy': 0,
        'surprised': 0,
        'angry': 0,
        'sad': 0,
        'fearful': 0,
        'disgusted': 0
    }
    
    # Ekstrak semua blendshapes ke dictionary untuk akses mudah
    blendshapes_dict = {bs.category_name: bs.score for bs in face_blendshapes}
    
    # Analisis ekspresi berdasarkan kombinasi blendshapes
    # 1. Senang (happy) - ditandai dengan senyum dan mata menyipit
    smile_score = (blendshapes_dict.get('mouthSmileLeft', 0) + blendshapes_dict.get('mouthSmileRight', 0)) / 2
    eye_squint_score = (blendshapes_dict.get('eyeSquintLeft', 0) + blendshapes_dict.get('eyeSquintRight', 0)) / 2
    expression_scores['happy'] = smile_score * 0.7 + eye_squint_score * 0.3
    
    # 2. Terkejut (surprised) - ditandai dengan mulut terbuka dan alis terangkat
    jaw_open_score = blendshapes_dict.get('jawOpen', 0)
    brow_up_score = (blendshapes_dict.get('browInnerUp', 0) + 
                   blendshapes_dict.get('browOuterUpLeft', 0) + 
                   blendshapes_dict.get('browOuterUpRight', 0)) / 3
    eye_wide_score = (blendshapes_dict.get('eyeWideLeft', 0) + blendshapes_dict.get('eyeWideRight', 0)) / 2
    expression_scores['surprised'] = jaw_open_score * 0.5 + brow_up_score * 0.3 + eye_wide_score * 0.2
    
    # 3. Marah (angry) - ditandai dengan alis menurun dan mulut mengerucut
    brow_down_score = (blendshapes_dict.get('browDownLeft', 0) + blendshapes_dict.get('browDownRight', 0)) / 2
    mouth_pucker_score = blendshapes_dict.get('mouthPucker', 0)
    expression_scores['angry'] = brow_down_score * 0.6 + mouth_pucker_score * 0.4
    
    # 4. Sedih (sad) - ditandai dengan sudut mulut menurun dan alis bagian dalam terangkat
    mouth_frown_score = (blendshapes_dict.get('mouthFrownLeft', 0) + blendshapes_dict.get('mouthFrownRight', 0)) / 2
    brow_inner_up_score = blendshapes_dict.get('browInnerUp', 0)
    expression_scores['sad'] = mouth_frown_score * 0.6 + brow_inner_up_score * 0.4
    
    # 5. Takut (fearful) - kombinasi terkejut dan cemas
    expression_scores['fearful'] = expression_scores['surprised'] * 0.6 + brow_down_score * 0.4
    
    # 6. Jijik (disgusted) - ditandai dengan hidung mengerut dan mulut atas terangkat
    nose_sneer_score = (blendshapes_dict.get('noseSneerLeft', 0) + blendshapes_dict.get('noseSneerRight', 0)) / 2
    mouth_upper_up_score = (blendshapes_dict.get('mouthUpperUpLeft', 0) + blendshapes_dict.get('mouthUpperUpRight', 0)) / 2
    expression_scores['disgusted'] = nose_sneer_score * 0.7 + mouth_upper_up_score * 0.3
    
    # # 7. Netral - dihitung sebagai kebalikan dari ekspresi lainnya
    # max_other_expressions = max(expression_scores.values())
    # expression_scores['neutral'] = max(0, 1 - max_other_expressions)
    
    # Debug: Cetak skor ekspresi
    print("Raw expression scores:", expression_scores)
    
    # Terjemahkan ke bahasa Indonesia
    translated_expressions = {
        # 'neutral': 'Netral',
        'happy': 'Senang',
        'surprised': 'Terkejut',
        'angry': 'Marah',
        'sad': 'Sedih',
        'fearful': 'Takut',
        'disgusted': 'Jijik'
    }
    
    # Dapatkan ekspresi dominan
    dominant_expression = max(expression_scores, key=expression_scores.get)
    
    # Normalisasi skor untuk presentasi (0-100%)
    total = sum(expression_scores.values())
    normalized_scores = {k: v/total for k, v in expression_scores.items()}
    
    return {
        'expression': translated_expressions[dominant_expression],
        'scores': {translated_expressions[k]: round(v * 100, 1) for k, v in normalized_scores.items()}
    }

@app.route('/detect_expression', methods=['POST'])
def detect_expression():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Baca dan konversi gambar
        image_data = file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Konversi ke format MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        with FaceLandmarker.create_from_options(options) as landmarker:
            detection_result = landmarker.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return jsonify({'error': 'Tidak terdeteksi wajah'}), 400
                
            # Proses ekspresi
            expression_result = process_expression(detection_result.face_blendshapes[0])
            
            # Gambar landmark di wajah
            annotated_image = image.copy()
            for landmark in detection_result.face_landmarks:
                for point in landmark:
                    x = int(point.x * image.shape[1])
                    y = int(point.y * image.shape[0])
                    cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)
            
            # Simpan gambar hasil
            result_filename = f"result_{str(uuid.uuid4())[:8]}.jpg"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, annotated_image)

            # print("Expression detected:", expression_result['scores'])
            
            return jsonify({
                'expression': expression_result['expression'],
                'scores': expression_result['scores'],
                'result_image_url': f'/uploads/{result_filename}'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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