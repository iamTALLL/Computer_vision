from flask import Flask, jsonify, request, send_file, render_template
# Giả định các hàm này được import từ processes.py
from processes import (
    KNOWLEDGE_DB, 
    log_transform, 
    power_law_transform, 
    negative_image, 
    equalize_histogram, 
    frequency_filter,
    convert_to_png, 
    spatial_filter,
    contra_harmonic_mean_filter,
    adaptive_local_filter,
    inverse_filter,
    otsu_segmentation,
    ml_segmentation
)
from io import BytesIO

app = Flask(__name__)

# --- HÀM HỖ TRỢ CHUNG ---
def get_image_bytes():
    if 'image_file' not in request.files:
        raise ValueError("Không tìm thấy file ảnh (image_file).")
    return request.files['image_file'].read()



@app.route('/', methods=['GET'])
def index():
    """Trả về trang giao diện chính (index.html)."""
    return render_template('index.html') # Flask sẽ tìm file trong thư mục templates

@app.route('/api/convert/to_png', methods=['POST'])
def convert_and_show_original():
    """
    Nhận file ảnh (có thể là .tif) và trả về nó dưới dạng PNG.
    """
    try:
        # Lấy file ảnh gốc
        image_bytes = get_image_bytes()
        
        # Chuyển đổi sang PNG
        processed_bytes = convert_to_png(image_bytes)
        
        # Trả về file PNG
        return send_file(
            BytesIO(processed_bytes),
            mimetype='image/png'
        )
    except Exception as e:
        return jsonify({"error": f"Lỗi chuyển đổi ảnh gốc: {str(e)}"}), 500

# --- ENDPOINT KIẾN THỨC CHUNG (GET) ---
@app.route('/api/knowledge/<tech_key>', methods=['GET'])
def get_knowledge(tech_key):
    knowledge = KNOWLEDGE_DB.get(tech_key)
    if knowledge:
        return jsonify(knowledge)
    return jsonify({"error": f"Không tìm thấy kiến thức cho thủ thuật: {tech_key}"}), 404

# -----------------------------------------------------
# --- TAB 1: CẢI THIỆN ẢNH ---
# -----------------------------------------------------

# 1. Biến đổi Âm bản
@app.route('/api/process/negative_image', methods=['POST'])
def process_negative_image():
    try:
        image_bytes = get_image_bytes()
        processed_bytes = negative_image(image_bytes) 
        return jsonify(processed_bytes)
    except Exception as e:
        return jsonify({"error": f"Lỗi Âm bản: {str(e)}"}), 500

# 2. Biến đổi Logarit
@app.route('/api/process/log_transform', methods=['POST'])
def process_log_transform():
    try:
        image_bytes = get_image_bytes()
        c = float(request.form.get('c', 1.0))
        processed_bytes = log_transform(image_bytes, c) 
        return jsonify(processed_bytes)
    except Exception as e:
        return jsonify({"error": f"Lỗi Logarit: {str(e)}"}), 500

# 3. Biến đổi Luật Công suất (Gamma)
@app.route('/api/process/power_law_transform', methods=['POST'])
def process_power_law_transform():
    try:
        image_bytes = get_image_bytes()
        gamma = float(request.form.get('gamma', 1.0))
        c = float(request.form.get('c', 1.0))
        processed_bytes = power_law_transform(image_bytes, c, gamma)
        return jsonify(processed_bytes)
    except Exception as e:
        return jsonify({"error": f"Lỗi Gamma: {str(e)}"}), 500

# 4. Cân bằng Histogram
@app.route('/api/process/histogram_equalization', methods=['POST'])
def process_histogram_equalization():
    try:
        image_bytes = get_image_bytes()
        processed_bytes = equalize_histogram(image_bytes)
        return jsonify(processed_bytes)
    except Exception as e:
        return jsonify({"error": f"Lỗi Histogram: {str(e)}"}), 500

# 5. Lọc miền không gian
@app.route('/api/process/spatial_filter', methods=['POST'])
def process_spatial_filter():
    try:
        image_bytes = get_image_bytes()
        
        # Lấy loại bộ lọc và kích thước kernel từ form data
        filter_type = request.form.get('filter_type', 'mean') # mean, median, laplacian_sharpen
        kernel_size = int(request.form.get('kernel_size', 3))
        
        # Gọi hàm xử lý chung
        processed_bytes = spatial_filter(image_bytes, filter_type, kernel_size)
        
        return jsonify(processed_bytes)
    except Exception as e:
        return jsonify({"error": f"Lỗi Lọc Miền Không gian: {str(e)}"}), 500

# -----------------------------------------------------
# --- TAB 2: LỌC MIỀN TẦN SỐ (Chọn lọc Tần số) ---
# -----------------------------------------------------

# 1. Bộ lọc Thông thấp Gaussian (GLPF)
@app.route('/api/process/gaussian_lowpass_filter', methods=['POST'])
def process_gaussian_lowpass_filter():
    try:
        image_bytes = get_image_bytes()
        D0 = int(request.form.get('D0', 50)) 
        
        # Chỉ truyền D0, các tham số khác dùng mặc định: type='gaussian', mode='lowpass'
        result_dict = frequency_filter(image_bytes, D0=D0) 
        
        return jsonify(result_dict)
    except Exception as e:
        # Sửa lại tên lỗi cho chính xác
        return jsonify({"error": f"Lỗi GLPF: {str(e)}"}), 500

# 2. Bộ lọc thông thấp Lý tưởng (ILPF)
@app.route('/api/process/ideal_lowpass_filter', methods=['POST'])
def process_ideal_lowpass_filter():
    try:
        image_bytes = get_image_bytes()
        D0 = int(request.form.get('D0', 50)) 
        
        # Thêm filter_type='ideal'
        result_dict = frequency_filter(image_bytes, D0=D0, filter_type='ideal') 
        
        return jsonify(result_dict)
    except Exception as e:
        # Sửa lại tên lỗi
        return jsonify({"error": f"Lỗi ILPF: {str(e)}"}), 500
    
# 3. Bộ lọc thông thấp ButterWorth (BLPF)
@app.route('/api/process/butterworth_lowpass_filter', methods=['POST'])
def process_butterworth_lowpass_filter():
    try:
        image_bytes = get_image_bytes()
        D0 = int(request.form.get('D0', 50)) 
        
        # Thêm filter_type='butterworth'
        result_dict = frequency_filter(image_bytes, D0=D0, filter_type='butterworth') 
        
        return jsonify(result_dict)
    except Exception as e:
        # Sửa lại tên lỗi
        return jsonify({"error": f"Lỗi BLPF: {str(e)}"}), 500

# 4. Bộ lọc Thông cao Gaussian (GHPF)
@app.route('/api/process/gaussian_highpass_filter', methods=['POST'])
def process_gaussian_highpass_filter():
    try:
        image_bytes = get_image_bytes()
        D0 = int(request.form.get('D0', 50))
        
        # Thêm filter_mode='highpass' (filter_type mặc định là 'gaussian')
        result_dict = frequency_filter(image_bytes, D0=D0, filter_mode='highpass') 
        
        return jsonify(result_dict)
    except Exception as e:
        return jsonify({"error": f"Lỗi GHPF: {str(e)}"}), 500
    
# 5. Bộ lọc Thông cao Lý tưởng (IHPF)
@app.route('/api/process/ideal_highpass_filter', methods=['POST'])
def process_ideal_highpass_filter():
    try:
        image_bytes = get_image_bytes()
        D0 = int(request.form.get('D0', 50))
        
        # Thêm filter_type='ideal' và filter_mode='highpass'
        result_dict = frequency_filter(image_bytes, D0=D0, filter_type='ideal', filter_mode='highpass') 
        
        return jsonify(result_dict)
    except Exception as e:
        # Sửa lại tên lỗi
        return jsonify({"error": f"Lỗi IHPF: {str(e)}"}), 500

# 6. Bộ lọc Thông cao ButterWorth (BHPF)
# ĐỔI TÊN HÀM TỪ process_gaussian_highpass_filter thành process_butterworth_highpass_filter
@app.route('/api/process/butterworth_highpass_filter', methods=['POST'])
def process_butterworth_highpass_filter(): 
    try:
        image_bytes = get_image_bytes()
        D0 = int(request.form.get('D0', 50))
        
        # Thêm filter_type='butterworth' và filter_mode='highpass'
        result_dict = frequency_filter(image_bytes, D0=D0, filter_type='butterworth', filter_mode='highpass') 
        
        return jsonify(result_dict)
    except Exception as e:
        # Sửa lại tên lỗi
        return jsonify({"error": f"Lỗi BHPF: {str(e)}"}), 500


# -----------------------------------------------------
# --- TAB 3: XỬ LÝ NÂNG CAO VÀ PHỤC HỒI ---
# -----------------------------------------------------

# 1. Bộ lọc trung bình nghịch điều hoà
@app.route('/api/process/contra_harmonic_mean', methods=['POST'])
def process_contra_harmonic_mean():
    try:
        image_bytes = get_image_bytes()
        
        # Tham số cần thiết: Q và kernel_size
        Q = float(request.form.get('Q', 1.5)) 
        kernel_size = int(request.form.get('kernel_size', 3))
        
        results_dict = contra_harmonic_mean_filter(image_bytes, kernel_size, Q)
        
        # Trả về JSON chứa Base64 strings
        return jsonify(results_dict) 
        
    except Exception as e:
        return jsonify({"error": f"Lỗi Contra-Harmonic Mean: {str(e)}"}), 500

# 2. Bộ lọc giảm nhiễu cục bộ thích nghi
@app.route('/api/process/adaptive_local_filter', methods=['POST'])
def process_adaptive_local_filter(): # Đổi tên hàm thành process_adaptive_local_filter
    try:
        image_bytes = get_image_bytes()
        
        # 1. Lấy Tham số Lọc (kernel_size)
        kernel_size = int(request.form.get('kernel_size', 3))
        
        # 2. Lấy Tham số Ước lượng Nhiễu (Noise Estimation Parameters)
        # Sửa lỗi cú pháp: Lấy giá trị từ form, sau đó chuyển sang int
        x_start = int(request.form.get('x_start', 0))
        y_start = int(request.form.get('y_start', 0))
        width = int(request.form.get('width', 10))
        height = int(request.form.get('height', 10))
        
        # 3. Gọi hàm xử lý (adaptive_local_filter)
        results_dict = adaptive_local_filter(
            image_bytes, 
            kernel_size, 
            x_start, 
            y_start, 
            width, 
            height
        )
        
        # Trả về JSON chứa Base64 strings
        return jsonify(results_dict) 
        
    except Exception as e:
        # Sửa lại thông báo lỗi cho Bộ lọc Thích nghi
        return jsonify({"error": f"Lỗi Bộ lọc Thích nghi: {str(e)}"}), 500

# 3. Bộ lọc inverse
@app.route('/api/process/inverse_filter', methods=['POST'])
def process_inverse_filter():
    try:
        image_bytes = get_image_bytes()
        
        # Tham số cần thiết: model_K và cutoff ratio
        model_k = float(request.form.get('modelK', 1.5)) 
        cutoff_ratio = float(request.form.get('cutoff_ratio'))
        
        results_dict = inverse_filter(image_bytes, model_k, cutoff_ratio)
        
        # Trả về JSON chứa Base64 strings
        return jsonify(results_dict) 
        
    except Exception as e:
        return jsonify({"error": f"Lỗi Inverse Filter: {str(e)}"}), 500

# -----------------------------------------------------
# --- TAB 3: PHÂN ĐOẠN ẢNH ---
# -----------------------------------------------------

# 1. Ngưỡng hoá Otsu
@app.route('/api/process/otsu_segmentation', methods=['POST'])
def process_otsu_segmentation():
    try:
        image_bytes = get_image_bytes()
        
        results_dict = otsu_segmentation(image_bytes)
        
        # Trả về JSON chứa Base64 strings
        return jsonify(results_dict) 
        
    except Exception as e:
        return jsonify({"error": f"Lỗi Inverse Filter: {str(e)}"}), 500

#2. Phân vùng sử dụng học máy
@app.route('/api/process/ml_segmentation', methods=['POST'])
def process_ml_segmentation():
    try:
        image_bytes = get_image_bytes()
        
        # BƯỚC 1: Lấy Loại Mô hình và Khởi tạo tham số
        model_type = request.form.get('model_type', 'kmeans')
        n_clusters = None
        final_bandwidth = None

        # BƯỚC 2: RẼ NHÁNH DỰA TRÊN MODEL_TYPE
        if model_type == 'kmeans':
            # K-MEANS cần n_clusters (Số nhóm K)
            n_clusters = int(request.form.get('n_clusters', 3))
            if n_clusters < 1: n_clusters = 3 # Đảm bảo K >= 1
            
        elif model_type == 'mean_shift':
            # MEAN SHIFT cần bandwidth
            bandwidth = float(request.form.get('bandwidth', 0.0))
            # Nếu bandwidth = 0 (hoặc rất nhỏ), chuyển thành None để MeanShift tự ước lượng.
            final_bandwidth = bandwidth if bandwidth > 1e-6 else None 
            
        else:
            raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")

        # BƯỚC 3: Gọi hàm xử lý chính
        # Chúng ta truyền tất cả các tham số (n_clusters=None hoặc bandwidth=None) 
        # và hàm ml_segmentation sẽ chỉ sử dụng tham số liên quan đến model_type.
        results_dict = ml_segmentation(
            image_bytes, 
            model_type=model_type, 
            n_clusters=n_clusters,
            bandwidth=final_bandwidth 
        )
        
        # Trả về JSON chứa Base64 strings
        return jsonify(results_dict) 
        
    except NotImplementedError as e:
        return jsonify({"error": f"Lỗi triển khai: {str(e)}"}), 501
    except Exception as e:
        return jsonify({"error": f"Lỗi Phân vùng ML: {str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)