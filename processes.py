import cv2
import numpy as np
from io import BytesIO
import base64
from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Trong processes.py

def convert_to_png(image_bytes):
    """
    Đọc dữ liệu ảnh bất kỳ và mã hóa nó thành PNG.
    Đây là hàm để hiển thị ảnh gốc trong trình duyệt.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Dùng IMREAD_UNCHANGED để đọc tất cả các kênh và độ sâu
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) 

    if img is None:
        raise ValueError("Không thể đọc được dữ liệu ảnh gốc.")

    # Nếu là ảnh có nhiều hơn 3 kênh (ví dụ: TIFF có kênh alpha hoặc ảnh khoa học)
    # bạn có thể cần xử lý thêm. Ở đây, ta chỉ cần đảm bảo nó được mã hóa lại.
    
    # Mã hóa ảnh sang PNG để trình duyệt có thể hiển thị
    is_success, buffer = cv2.imencode(".png", img)
    if not is_success:
        raise RuntimeError("Lỗi khi mã hóa ảnh gốc sang PNG.")

    return BytesIO(buffer).read()

def encode_image_to_base64(img_array):
    """Mã hóa mảng NumPy (uint8) thành chuỗi Base64."""
    is_success, buffer = cv2.imencode(".png", img_array)
    if not is_success:
        raise RuntimeError("Lỗi mã hóa Base64.")
    return base64.b64encode(buffer).decode('utf-8')

def log_transform(image_bytes, c=1.0):
    """
    Thực hiện Biến đổi Logarit trên ảnh.
    Đầu vào: bytes của ảnh, hằng số c.
    Đầu ra: bytes của ảnh đã qua xử lý.
    """
    
    # 1. Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # Đọc ảnh xám
    
    if img is None:
        raise ValueError("Không thể đọc được dữ liệu ảnh.")

    # 2. Chuyển đổi kiểu dữ liệu (từ uint8 sang float)
    # Rất quan trọng: Logarit hoạt động trên float
    img_float = img.astype(np.float32)

    # 3. Áp dụng công thức Logarit: s = c * log(1 + r)
    # NumPy.log() tính logarit tự nhiên (ln)
    transformed_img_float = c * np.log(1 + img_float)

    # 4. Chuẩn hóa và chuyển lại về kiểu dữ liệu 8-bit (uint8)
    # Tìm giá trị max để chuẩn hóa về dải 0-255
    max_val = np.max(transformed_img_float)
    
    # Chuẩn hóa về dải 0-255
    if max_val > 0:
        normalized_img = (transformed_img_float / max_val) * 255
    else:
        normalized_img = transformed_img_float 
    
    # Chuyển về kiểu uint8 và làm tròn
    final_img = normalized_img.astype(np.uint8)
    processed_base64 = encode_image_to_base64(final_img)
    
    # Trả về dict, tương thích với JSON output
    return {
        'filtered_image': processed_base64,
        # Các thủ thuật Tab 1 không có phổ, chỉ trả về ảnh đã xử lý
    }

def power_law_transform(image_bytes, c=1.0, gamma=0.6):
    """
    Thực hiện Biến đổi Power Law (Gamma Correction) trên ảnh.
    Đầu vào: bytes của ảnh, hằng số c, hằng số gamma.
    Đầu ra: bytes của ảnh đã qua xử lý.
    
    Lưu ý: Công thức chuẩn thường là s = c * (r/L-1)^gamma * (L-1)
    """
    
    # 1. Đọc ảnh từ bytes (chỉ xử lý ảnh xám)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Không thể đọc được dữ liệu ảnh.")

    # 2. Chuẩn hóa Input và Áp dụng Công thức
    
    # Chuyển đổi sang float32
    img_float = img.astype(np.float32)
    
    # Chuẩn hóa về dải [0, 1] trước khi áp dụng luật lũy thừa
    # Giá trị max của uint8 là 255.
    img_normalized = img_float / 255.0

    # Áp dụng công thức Power Law: s = c * r^gamma
    transformed_img_float = c * np.power(img_normalized, gamma)

    # 3. Chuyển kết quả trở lại dải [0, 255]
    # Nhân kết quả đã chuẩn hóa với 255
    final_img_float = transformed_img_float * 255.0
    
    # Đảm bảo giá trị không vượt quá dải [0, 255] và chuyển về uint8
    # Sử dụng np.clip để giới hạn giá trị và .astype(np.uint8)
    final_img = np.clip(final_img_float, 0, 255).astype(np.uint8)
    processed_base64 = encode_image_to_base64(final_img)
    
    # Trả về dict, tương thích với JSON output
    return {
        'filtered_image': processed_base64,
        # Các thủ thuật Tab 1 không có phổ, chỉ trả về ảnh đã xử lý
    }

def negative_image(image_bytes):
    """
    Thực hiện Biến đổi Images Negative trên ảnh.
    Đầu vào: bytes của ảnh
    Đầu ra: bytes của ảnh đã qua xử lý.
    
    Lưu ý: Công thức chuẩn thường là $s = L - 1 - r$ (với $L=256$ cho ảnh 8-bit, $s = 255 - r$).
    """
    
    # 1. Đọc ảnh từ bytes (chỉ xử lý ảnh xám)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Không thể đọc được dữ liệu ảnh.")

    # 2. Chuẩn hóa Input và Áp dụng Công thức
    
    # Chuyển đổi sang float32
    img_float = img.astype(np.float32)
    

    # Áp dụng công thức 
    transformed_img_float = 255 - img_float

    # 3. Trả về 
    final_img = np.clip(transformed_img_float, 0, 255).astype(np.uint8)
    processed_base64 = encode_image_to_base64(final_img)
    
    # Trả về dict, tương thích với JSON output
    return {
        'filtered_image': processed_base64,
        # Các thủ thuật Tab 1 không có phổ, chỉ trả về ảnh đã xử lý
    }

def equalize_histogram(image_bytes):
    """
    Thực hiện Cân bằng Histogram cho ảnh xám.
    Đầu vào: Dữ liệu bytes của ảnh.
    Đầu ra: Dữ liệu bytes của ảnh đã qua xử lý.
    """
    # 1. Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Không thể đọc được dữ liệu ảnh.")

    # 2. Chuyển sang ảnh xám để cân bằng (hoặc YUV cho ảnh màu)
    if img.ndim == 3 and img.shape[2] >= 3:
        # --- RẼ NHÁNH 1: XỬ LÝ ẢNH MÀU ---
        
        # 1. Chuyển sang YUV (hoặc HSV)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        # 2. Lấy kênh Cường độ (Y)
        Y_channel = img_yuv[:,:,0] 
        
        # 3. Áp dụng Thủ thuật (Ví dụ: Cân bằng Histogram)
        processed_Y = cv2.equalizeHist(Y_channel)
        
        # 4. Hợp nhất lại và chuyển về BGR
        img_yuv[:,:,0] = processed_Y
        final_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
    else:
        # --- RẼ NHÁNH 2: XỬ LÝ ẢNH XÁM ---
        
        # Đảm bảo ảnh là ảnh xám 1 kênh (nếu ban đầu là màu, cần chuyển)
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        # 3. Áp dụng Thủ thuật trực tiếp (Ví dụ: Cân bằng Histogram)
        final_img = cv2.equalizeHist(img_gray)
        
    processed_base64 = encode_image_to_base64(final_img)
    
    # Trả về dict, tương thích với JSON output
    return {
        'filtered_image': processed_base64,
        # Các thủ thuật Tab 1 không có phổ, chỉ trả về ảnh đã xử lý
    }




def apply_spatial_filter(img_channel, filter_type, kernel_size):
    """
    Áp dụng bộ lọc không gian (Mean, Median, hoặc Laplacian) lên một kênh ảnh.
    Đầu vào: Kênh ảnh (1D numpy array), loại bộ lọc, kích thước kernel.
    Đầu ra: Kênh ảnh đã xử lý.
    """
    if filter_type == 'mean':
        # Lọc Trung bình (Làm mượt tuyến tính)
        # cv2.blur chỉ yêu cầu kích thước kernel (width, height)
        # Kích thước kernel phải là số lẻ cho các bộ lọc thông thường, nhưng blur chấp nhận cả chẵn
        processed_channel = cv2.blur(img_channel, (kernel_size, kernel_size))
        
    elif filter_type == 'median':
        # Lọc Trung vị (Khử nhiễu phi tuyến tính)
        # Kích thước kernel phải là SỐ LẺ
        if kernel_size % 2 == 0:
            kernel_size += 1 # Đảm bảo kernel_size là số lẻ
        processed_channel = cv2.medianBlur(img_channel, kernel_size)
        
    elif filter_type == 'laplacian_sharpen':
        # Làm sắc nét (Sharpening) bằng Laplacian
        
        # 1. Tính Laplacian (Phát hiện cạnh)
        # cv2.Laplacian chỉ nhận kích thước kernel lẻ (vd: 3, 5). Kernel mặc định là 1.
        laplacian_img = cv2.Laplacian(img_channel, cv2.CV_64F, ksize=kernel_size)
        
        # 2. Làm sắc nét bằng cách kết hợp: sharpened = original - Laplacian
        # Chuyển ảnh gốc sang float64 để tính toán
        img_float = img_channel.astype(np.float64)
        sharpened_float = img_float - laplacian_img
        
        # 3. Chuẩn hóa về dải 0-255 và chuyển về uint8
        # Đảm bảo giá trị không vượt quá 0-255
        processed_channel = np.clip(sharpened_float, 0, 255).astype(np.uint8)
        
    else:
        raise ValueError(f"Loại bộ lọc không gian không hợp lệ: {filter_type}")
        
    return processed_channel


def spatial_filter(image_bytes, filter_type='mean', kernel_size=3):
    """
    Hàm chung xử lý Lọc Miền Không gian cho cả ảnh xám và ảnh màu.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Không thể đọc được dữ liệu ảnh.")
    
    # 1. XỬ LÝ ẢNH MÀU (3 Kênh)
    if img.ndim == 3 and img.shape[2] >= 3:
        # Chuyển sang YUV (hoặc HSV)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        # Lấy kênh Y (Cường độ)
        Y_channel = img_yuv[:,:,0] 
        
        # Áp dụng bộ lọc lên kênh Y
        processed_Y = apply_spatial_filter(Y_channel, filter_type, kernel_size)
        
        # Hợp nhất lại
        img_yuv[:,:,0] = processed_Y
        final_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
    # 2. XỬ LÝ ẢNH XÁM (1 Kênh)
    else:
        # Đảm bảo ảnh là ảnh xám 1 kênh
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        final_img = apply_spatial_filter(img_gray, filter_type, kernel_size)
        
    # 3. Mã hóa ảnh kết quả về bytes để trả về
    processed_base64 = encode_image_to_base64(final_img)
    
    # Trả về dict, tương thích với JSON output
    return {
        'filtered_image': processed_base64,
        # Các thủ thuật Tab 1 không có phổ, chỉ trả về ảnh đã xử lý
    }

def create_magnitude_spectrum(dft_shifted):
    """
    Tính Phổ Biên độ (Magnitude Spectrum) từ DFT đã dịch chuyển (Shifted DFT).
    Áp dụng log transform để dễ nhìn hơn.
    """
    # dft_shifted có 2 kênh (Real và Imaginary)
    magnitude_spectrum = cv2.magnitude(dft_shifted[:,:,0], dft_shifted[:,:,1])
    
    # Áp dụng log transform để nén dải động
    # log(1 + magnitude)
    magnitude_spectrum += 1 # tránh log(0)
    magnitude_spectrum = np.log(magnitude_spectrum)
    
    # Chuẩn hóa về dải 0-255 và chuyển về uint8
    min_val = np.min(magnitude_spectrum)
    max_val = np.max(magnitude_spectrum)
    
    if max_val - min_val > 1e-6:
        normalized = 255 * (magnitude_spectrum - min_val) / (max_val - min_val)
    else:
        normalized = magnitude_spectrum # Trường hợp ảnh đen tuyệt đối

    return normalized.astype(np.uint8)

def encode_image_to_base64(img_array):
    """Mã hóa mảng NumPy (uint8) thành chuỗi Base64."""
    is_success, buffer = cv2.imencode(".png", img_array)
    if not is_success:
        raise RuntimeError("Lỗi mã hóa Base64.")
    # Trả về chuỗi Base64 (bỏ phần header bytes)
    return base64.b64encode(buffer).decode('utf-8')

def create_frequency_filter_mask(shape, D0, filter_type='gaussian', n=2, filter_mode='lowpass'):
    """
    Tạo Mask bộ lọc tần số đa năng (Gaussian, Butterworth, Ideal)
    shape: Kích thước ảnh (M, N).
    D0: Tần số cắt (cutoff frequency).
    filter_type: 'gaussian', 'butterworth', 'ideal'.
    n: Bậc của bộ lọc Butterworth.
    filter_mode: 'lowpass' hoặc 'highpass'.
    """
    M, N = shape
    
    # 1. Tính toán Ma trận Khoảng cách D(u,v)
    u = np.arange(M)
    v = np.arange(N)
    u[M//2:] = u[M//2:] - M
    v[N//2:] = v[N//2:] - N
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2) # D(u,v)

    # 2. Tính Hàm truyền H_lowpass (H_lp)
    if filter_type == 'ideal':
        # Ideal LPF (ILPF)
        H_lp = np.where(D <= D0, 1, 0) # H(u,v) = 1 nếu D <= D0, ngược lại là 0
    
    elif filter_type == 'butterworth':
        # Butterworth LPF (BLPF)
        # H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
        H_lp = 1.0 / (1.0 + (D / D0)**(2 * n))
        
    elif filter_type == 'gaussian':
        # Gaussian LPF (GLPF)
        # H(u,v) = e^(-D^2(u,v) / 2*D0^2)
        H_lp = np.exp(-(D**2) / (2 * D0**2))
    
    else:
        raise ValueError("Loại bộ lọc không hợp lệ. Chọn 'gaussian', 'butterworth', hoặc 'ideal'.")

    # 3. Tính Hàm truyền Cuối cùng (Thông thấp hoặc Thông cao)
    if filter_mode == 'lowpass':
        return H_lp
    elif filter_mode == 'highpass':
        # H_hp = 1 - H_lp
        return 1 - H_lp
    else:
        raise ValueError("Chế độ lọc không hợp lệ. Chọn 'lowpass' hoặc 'highpass'.")
    
# Sửa tên hàm trong processes.py thành hàm chung
def frequency_filter(image_bytes, filter_type='gaussian', filter_mode='lowpass', D0=50, n=2):
    """
    Hàm chung áp dụng bất kỳ Bộ lọc Tần số nào (GLPF, BLPF, ILPF, GHPF, BHPF, IHPF).
    """
    # 1. Đọc và chuẩn bị ảnh (Giữ nguyên logic của hàm cũ)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Đọc ảnh màu
    
    if img_bgr is None: raise ValueError("Không thể đọc được dữ liệu ảnh.")

    # KIỂM TRA: Nếu là ảnh màu, ta chỉ lọc kênh Y (Luminance)
    is_color = (img_bgr.ndim == 3 and img_bgr.shape[2] >= 3)
    
    if is_color:
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        img_float = np.float32(img_yuv[:, :, 0]) # Lấy kênh Y (Cường độ)
    else:
        # Nếu là ảnh xám (hoặc 1 kênh), chuyển về xám để xử lý
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_float = np.float32(img_gray)
    
    M, N = img_float.shape

    # 2. Biến đổi Fourier Rời rạc (DFT) và Shift
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    dft_shifted_original = dft_shifted.copy()
    
    # 3. TẠO VÀ ÁP DỤNG BỘ LỌC (Sử dụng hàm đa năng mới)
    H = create_frequency_filter_mask((M, N), D0, filter_type, n, filter_mode)
    H_expanded = np.stack([H, H], axis=-1) 
    
    # G(u,v) = F(u,v) * H(u,v)
    G_shifted = dft_shifted * H_expanded

    # 4. Biến đổi Fourier Ngược (IDFT) 
    G = np.fft.ifftshift(G_shifted)
    img_back = cv2.idft(G, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    # 5. Chuẩn hóa và chuyển về uint8 (Cần thiết cho cả LPF và HPF)
    min_val = np.min(img_back)
    max_val = np.max(img_back)
    
    # Scaling về 0-255
    normalized_img = 255 * (img_back - min_val) / (max_val - min_val + 1e-6) 
    processed_channel = np.clip(normalized_img, 0, 255).astype(np.uint8)  
    
    if is_color:
        # Thay thế kênh Y đã lọc vào ảnh YUV gốc
        img_yuv[:, :, 0] = processed_channel
        final_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        final_img = processed_channel # Ảnh xám  
    # Phổ 1: Phổ GỐC |F(u,v)|
    spectrum_original = create_magnitude_spectrum(dft_shifted_original)
    
    # Phổ 2: Phổ ĐÃ LỌC |G(u,v)|
    spectrum_filtered = create_magnitude_spectrum(G_shifted)

    # Ảnh 3: Mask H(u,v) (Chuẩn hóa để hiển thị)
    mask_H_vis = (H * 255).astype(np.uint8)
    
    # 5. MÃ HÓA VÀ TRẢ VỀ DƯỚI DẠNG JSON
    results = {
        'filtered_image': encode_image_to_base64(final_img),
        'original_spectrum': encode_image_to_base64(spectrum_original),
        'filtered_spectrum': encode_image_to_base64(spectrum_filtered),
        'filter_mask': encode_image_to_base64(mask_H_vis)
    }

    return results

def contra_harmonic_mean_filter(image_bytes, kernel_size=3, Q=1.5):
    """
    Áp dụng Bộ lọc Trung bình Nghịch điều hòa.
    Q > 0: Loại bỏ nhiễu tiêu (Pepper Noise)
    Q < 0: Loại bỏ nhiễu muối (Salt Noise)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Không thể đọc được dữ liệu ảnh.")

    is_color = (img_bgr.ndim == 3 and img_bgr.shape[2] >= 3)
    
    # 1. Chuẩn bị kênh cường độ (Intensity Channel)
    if is_color:
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        img_intensity = img_yuv[:, :, 0] # Kênh Y
    else:
        img_intensity = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Chuyển đổi sang float để tính lũy thừa
    img_float = img_intensity.astype(np.float64) 
    
    # Kích thước kernel phải là số lẻ
    ksize = kernel_size if kernel_size % 2 != 0 else kernel_size + 1 
    
    # 2. Tính lũy thừa Q và Q+1 (Tử số và Mẫu số)
    
    # TỬ SỐ: g(s,t)^(Q+1)
    numerator_power = img_float ** (Q + 1)
    
    # MẪU SỐ: g(s,t)^Q
    denominator_power = img_float ** Q
    
    # 3. Tính tổng cục bộ (Sử dụng cv2.boxFilter để tính tổng hiệu quả)
    # cv2.boxFilter(src, ddepth, ksize, normalize=False)
    # normalize=False: Hàm sẽ tính tổng (sum) trong cửa sổ kernel_size
    
    sum_numerator = cv2.boxFilter(numerator_power, -1, (ksize, ksize), normalize=False)
    sum_denominator = cv2.boxFilter(denominator_power, -1, (ksize, ksize), normalize=False)
    
    # 4. Áp dụng Công thức (Tránh chia cho 0)
    # Lọc nghịch điều hòa = Sum(Q+1) / Sum(Q)
    
    # Tạo mask để tránh chia cho giá trị 0 hoặc rất nhỏ (để ổn định)
    zero_mask = sum_denominator == 0
    
    # Thực hiện phép chia
    filtered_channel_float = np.divide(sum_numerator, sum_denominator, 
                                     out=np.zeros_like(img_float, dtype=np.float64), 
                                     where=~zero_mask)
                                     
    # 5. Chuẩn hóa và Hợp nhất
    
    # Đảm bảo dải cường độ nằm trong [0, 255]
    processed_channel = np.clip(filtered_channel_float, 0, 255).astype(np.uint8)

    if is_color:
        # Hợp nhất lại kênh Y đã xử lý
        img_yuv[:, :, 0] = processed_channel
        final_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        final_img = processed_channel

    # 6. Mã hóa và trả về JSON dict
    processed_base64 = encode_image_to_base64(final_img)
    return {
        'filtered_image': processed_base64,
    }


def estimate_noise_variance(img_array, x_start, y_start, width, height):
    """
    Ước lượng phương sai nhiễu (sigma_eta^2) từ một vùng đồng nhất của ảnh.

    Tham số:
        img_array: Mảng NumPy của ảnh (nên là ảnh xám).
        x_start, y_start: Tọa độ bắt đầu của vùng ước lượng (pixel).
        width, height: Kích thước của vùng ước lượng.

    Đầu ra:
        float: Phương sai nhiễu ước lượng (sigma_eta^2).
    """
    
    if img_array.ndim > 2:
        # Nếu là ảnh màu, chuyển sang ảnh xám (intensity) trước khi ước lượng
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
    # 1. Trích xuất vùng ảnh đồng nhất (strip)
    # Lấy vùng từ y_start đến y_start + height, và x_start đến x_start + width
    # Chú ý: NumPy sử dụng thứ tự (row, col) tương đương (y, x)
    
    try:
        # Lấy vùng ảnh
        strip = img_array[y_start:y_start + height, x_start:x_start + width]
    except IndexError:
        raise ValueError("Tọa độ vùng ước lượng nằm ngoài phạm vi ảnh.")
        
    # 2. Kiểm tra kích thước mẫu (Nếu quá nhỏ, kết quả không đáng tin cậy)
    if strip.size < 9: # Ví dụ: kích thước tối thiểu 3x3
        raise ValueError("Vùng ước lượng quá nhỏ để tính toán thống kê.")

    # 3. Tính Phương sai (Variance)
    # Trong vùng đồng nhất, phương sai cục bộ chính là phương sai nhiễu (sigma_L^2 ≈ sigma_eta^2)
    
    # Dùng np.var() để tính phương sai. Sử dụng dtype float64 cho độ chính xác cao
    noise_variance = np.var(strip.astype(np.float64))
    
    return noise_variance

def adaptive_local_filter(image_bytes, kernel_size=3, x_start=0, y_start=0, width=10, height=10):
    """
    Áp dụng Bộ lọc Giảm nhiễu Cục bộ Thích nghi.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None: raise ValueError("Không thể đọc được dữ liệu ảnh.")

    is_color = (img_bgr.ndim == 3 and img_bgr.shape[2] >= 3)
    
    # 1. Chuẩn bị kênh cường độ (Intensity Channel) và ƯỚC LƯỢNG PHƯƠNG SAI NHIỄU
    if is_color:
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        img_intensity = img_yuv[:, :, 0] # Kênh Y
    else:
        img_intensity = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Ước lượng Phương sai Nhiễu (σ_η^2) từ ảnh cường độ gốc (trước khi lọc)
    try:
        noise_variance = estimate_noise_variance(img_intensity, x_start, y_start, width, height)
    except ValueError as e:
        # Nếu ước lượng thất bại (vùng quá nhỏ hoặc ngoài biên), gán giá trị mặc định nhỏ
        print(f"Cảnh báo: {e}. Sử dụng phương sai nhiễu mặc định 10.0")
        noise_variance = 10.0 
        
    img_float = img_intensity.astype(np.float64) # Giữ kiểu float64
    ksize = kernel_size if kernel_size % 2 != 0 else kernel_size + 1 
    
    # 2. Tính Thống kê Cục bộ (Local Statistics)
    
    # Tính Trung bình Cục bộ (m_L)
    local_mean = cv2.boxFilter(img_float, -1, (ksize, ksize), normalize=True)
    
    # Tính Phương sai Cục bộ (σ_L^2)
    # E[g^2] = Trung bình cục bộ của ảnh bình phương
    E_g_sq = cv2.boxFilter(img_float**2, -1, (ksize, ksize), normalize=True)
    # m_L^2 = Bình phương của Trung bình cục bộ
    m_L_sq = local_mean**2
    
    local_variance = E_g_sq - m_L_sq
    # Phương sai không thể âm, cần clip về 0
    local_variance = np.clip(local_variance, 0, None)
    
    # 3. Áp dụng Công thức Bộ lọc Thích nghi
    
    # Tỷ lệ khuếch đại A = σ_η^2 / σ_L^2
    # Thêm EPSILON để ổn định (tránh chia cho 0)
    EPSILON = 1e-6
    
    # Chỉ áp dụng lọc khi phương sai cục bộ lớn hơn phương sai nhiễu
    # Nếu σ_L^2 < σ_η^2, ta coi vùng đó là nhiễu thuần túy.
    
    # Tính hệ số A = max(0, 1 - (σ_η^2 / σ_L^2))
    # Công thức gốc (đã được sắp xếp lại):
    # filtered = g(x,y) - (σ_η^2 / σ_L^2) * (g(x,y) - m_L)
    
    # Kiểm soát: Nếu σ_L^2 < σ_η^2, A sẽ > 1, làm khuếch đại nhiễu, đây là lỗi
    # Cần giới hạn tỷ lệ (σ_η^2 / σ_L^2) tối đa bằng 1
    # Công thức ổn định:
    A = noise_variance / (local_variance + EPSILON) # Tính tỷ lệ
    
    # Giới hạn tỷ lệ A tối đa bằng 1 (theo nguyên lý bộ lọc thích nghi)
    A = np.clip(A, 0, 1)
    
    # Áp dụng công thức
    filtered_channel_float = img_float - A * (img_float - local_mean)
    
    # 4. Chuẩn hóa và Hợp nhất
    processed_channel = np.clip(filtered_channel_float, 0, 255).astype(np.uint8)

    if is_color:
        img_yuv[:, :, 0] = processed_channel
        final_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:
        final_img = processed_channel

    # 5. Mã hóa và trả về JSON dict
    processed_base64 = encode_image_to_base64(final_img)
    return {'filtered_image': processed_base64}

def create_degradation_mask(shape, k):
    """
    Tạo Mask hàm suy giảm H(u,v) (Ví dụ: Mô hình Khí quyển)
    Công thức: H(u,v) = e^(-k * (u^2 + v^2)^(5/6))
    """
    M, N = shape
    
    # 1. Tính toán Ma trận Khoảng cách D_sq = (u^2 + v^2)
    u = np.arange(M)
    v = np.arange(N)
    u[M//2:] = u[M//2:] - M
    v[N//2:] = v[N//2:] - N
    V, U = np.meshgrid(v, u)
    D_sq = U**2 + V**2 

    # 2. Áp dụng công thức suy giảm (H(u,v))
    # Sử dụng np.power(D_sq, 5/6)
    H = np.exp(-k * np.power(D_sq, 5/6))
    
    # Clip H(u,v) để tránh chia cho giá trị 0 tuyệt đối 
    H = np.clip(H, 1e-6, None) 
    
    return H

def inverse_filter(image_bytes, model_k=0.0025, cutoff_ratio=0.7):
    """
    Áp dụng Lọc Nghịch đảo (Inverse Filtering) trong miền tần số.
    image_bytes: Ảnh bị suy giảm (ví dụ: bị mờ/nhiễu).
    model_k: Tham số cường độ suy giảm (cho mô hình H(u,v)).
    cutoff_ratio: Giới hạn tần số để tránh khuếch đại nhiễu (ví dụ: 0.7 = 70% phổ).
    """
    # 1. Chuẩn bị ảnh (Giả định ảnh xám để đơn giản hóa)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError("Không thể đọc được dữ liệu ảnh.")

    img_float = np.float32(img)
    M, N = img_float.shape

    # 2. DFT và Shift ảnh suy giảm G(u,v)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    G_shifted = np.fft.fftshift(dft)
    
    # 3. Ước lượng Hàm suy giảm H(u,v)
    H = create_degradation_mask((M, N), model_k)
    
    # 4. LỌC NGHỊCH ĐẢO (Inverse Filtering)
    
    # Tạo Mask giới hạn (Cutoff Mask) để loại bỏ các tần số cao
    # (nơi H(u,v) rất nhỏ và nhiễu N(u,v) bị khuếch đại)
    center_y, center_x = M // 2, N // 2
    y, x = np.ogrid[0:M, 0:N]
    
    # Khoảng cách từ tâm (để giới hạn tần số)
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_radius = np.max(radius)
    
    # Tạo mask hình tròn, giữ lại phần trung tâm (tần số thấp)
    cutoff_D = max_radius * cutoff_ratio 
    frequency_mask = (radius <= cutoff_D).astype(np.float32)
    
    # Mở rộng H và Mask cho mảng phức 2 kênh
    H_expanded = np.stack([H, H], axis=-1)
    Mask_expanded = np.stack([frequency_mask, frequency_mask], axis=-1)
    
    # Áp dụng Lọc Nghịch đảo: F_hat = G / H * Mask
    F_hat_shifted = np.divide(G_shifted, H_expanded, 
                              out=np.zeros_like(G_shifted), 
                              where=(H_expanded != 0))
                              
    # Áp dụng giới hạn tần số
    F_hat_shifted *= Mask_expanded

    # 5. IDFT và Chuẩn hóa (Giữ nguyên logic HPF/LPF)
    G = np.fft.ifftshift(F_hat_shifted)
    img_back = cv2.idft(G, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    # Chuẩn hóa về dải 0-255
    min_val, max_val = np.min(img_back), np.max(img_back)
    normalized_img = 255 * (img_back - min_val) / (max_val - min_val + 1e-6)
    final_img = np.clip(normalized_img, 0, 255).astype(np.uint8)

    # 6. Mã hóa và trả về JSON dict
    processed_base64 = encode_image_to_base64(final_img)
    
    # Cần trả về phổ để minh họa Mask và H(u,v)
    spectrum_original = create_magnitude_spectrum(G_shifted) # G(u,v) là ảnh suy giảm
    
    return {
        'filtered_image': processed_base64,
        'original_spectrum': encode_image_to_base64(spectrum_original),
        'filter_mask': encode_image_to_base64((H * 255).astype(np.uint8)), # H(u,v) Mask
        'filtered_spectrum': encode_image_to_base64(create_magnitude_spectrum(F_hat_shifted)) # Phổ đã phục hồi
    }
    
def otsu_segmentation(image_bytes):
    """
    Thực hiện ngưỡng hóa Otsu (Otsu Thresholding) tự động.
    Đầu ra: Ảnh nhị phân đã phân vùng.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None: raise ValueError("Không thể đọc được dữ liệu ảnh.")

    # 1. Chuyển sang ảnh xám (Bắt buộc cho Otsu)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Áp dụng Otsu Thresholding
    # cv2.THRESH_BINARY | cv2.THRESH_OTSU: kết hợp ngưỡng nhị phân và phương pháp Otsu
    # Otsu tự động tìm ngưỡng T tối ưu
    T, segmented_img = cv2.threshold(
        img_gray, 
        0, 
        255, 
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Trả về kết quả (ảnh nhị phân)
    processed_base64 = encode_image_to_base64(segmented_img)
    return {
        'filtered_image': processed_base64,
        'note': f"Ngưỡng Tối ưu tìm được (Otsu Threshold): {T}"
    }

def ml_segmentation(image_bytes, model_type='kmeans', n_clusters=3, bandwidth=None):
    """
    Phân vùng ảnh sử dụng mô hình học máy (K-Means/Mean Shift).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None: raise ValueError("Không thể đọc được dữ liệu ảnh.")
    
    # 1. Chuẩn bị dữ liệu (Reshape ảnh 2D thành mảng 1D của các pixel RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixel_data = img_rgb.reshape((-1, 3)).astype(np.float32)
    
    # Khởi tạo ảnh cuối cùng
    final_img = None
    note = "Phân vùng hoàn tất."

    # 2. ÁP DỤNG THUẬT TOÁN ML
    
    if model_type == 'kmeans':
        # --- K-MEANS (Sử dụng OpenCV) ---
        if n_clusters < 2 or n_clusters > 10:
             n_clusters = np.clip(n_clusters, 2, 10)
             note = f"Số nhóm K được giới hạn về {n_clusters}."
             
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixel_data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        centers = np.uint8(centers)
        segmented_img_rgb = centers[labels.flatten()]
        segmented_img_rgb = segmented_img_rgb.reshape(img_bgr.shape)
        final_img = cv2.cvtColor(segmented_img_rgb, cv2.COLOR_RGB2BGR)
        note = f"Phân vùng hoàn tất với {len(centers)} nhóm (K-Means)."

    elif model_type == 'mean_shift':
        
        if bandwidth is None or bandwidth <= 0:
            # Bandwidth: Khoảng cách mà các điểm lân cận ảnh hưởng đến một điểm
            # Nếu không có tham số bandwidth, MeanShift sẽ tự động ước lượng.
            # Ta có thể bỏ qua việc ước lượng thủ công nếu MeanShift tự xử lý.
            pass
        
        # 2. Khởi tạo và Fit mô hình
        # MeanShift không cần n_clusters
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1) 
        
        # Chuyển pixel_data về uint8 vì MeanShift yêu cầu int
        ms.fit(pixel_data.astype(np.uint8)) 
        
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        n_clusters_ms = len(np.unique(labels))
        
        # 3. Tái tạo ảnh phân vùng
        cluster_centers = np.uint8(cluster_centers)
        segmented_img_rgb = cluster_centers[labels]
        segmented_img_rgb = segmented_img_rgb.reshape(img_bgr.shape)
        
        final_img = cv2.cvtColor(segmented_img_rgb, cv2.COLOR_RGB2BGR)
        note = f"Phân vùng hoàn tất với {n_clusters_ms} nhóm (Mean Shift)."

    else:
        raise ValueError("Loại mô hình không được hỗ trợ.")

    # Mã hóa và trả về JSON dict
    processed_base64 = encode_image_to_base64(final_img)
    return {
        'filtered_image': processed_base64,
        'note': note
    }

# Dữ liệu kiến thức mô phỏng
KNOWLEDGE_DB = {
    "negative_image": {
        "name": "Biến đổi Âm bản (Image Negative)",
        "description": "Thủ thuật đảo ngược mức xám của ảnh. Mức sáng nhất trở thành tối nhất, và ngược lại. Hữu ích cho việc tăng cường chi tiết ở các vùng ảnh tối.",
        "formula": "Công thức biến đổi: $$s = L - 1 - r$$ Với ảnh 8-bit, $L=256$, công thức là $s = 255 - r$.",
        "purpose": "Tạo ảnh âm bản, làm nổi bật các chi tiết ở vùng tối (ví dụ: ảnh y tế)."
    },
    "log_transform": {
        "name": "Biến đổi Logarit (Logarithmic Transformation)",
        "description": "Nén dải động của ảnh, ánh xạ dải cường độ đầu vào hẹp sang dải cường độ đầu ra rộng hơn. Đặc biệt hiệu quả cho việc làm nổi bật chi tiết ở vùng tối của ảnh có dải động lớn.",
        "formula": "Công thức biến đổi: $$s = c \\cdot \\log(1 + r)$$",
        "purpose": "Nén dải động lớn (ví dụ: phổ Fourier, ảnh chụp X-quang) để các chi tiết tối có thể nhìn thấy."
    },
    "power_law_transform": {
        "name": "Biến đổi Luật Công suất (Gamma Correction)",
        "description": "Điều chỉnh độ sáng và độ tương phản phi tuyến tính. Nếu $\\gamma < 1$ làm sáng ảnh, $\\gamma > 1$ làm tối ảnh. Đây là phương pháp điều chỉnh gamma tiêu chuẩn.",
        "formula": "Công thức biến đổi: $$s = c \\cdot r^{\\gamma}$$ Trong đó, $r$ là cường độ pixel đã được chuẩn hóa về dải [0, 1].",
        "purpose": "Hiệu chỉnh độ sáng màn hình (gamma correction) hoặc cải thiện chi tiết vùng tối/sáng."
    },
    "mean_filter": {
        "name": "Lọc Trung bình (Smoothing Linear Filter)",
        "description": "Đây là bộ lọc tuyến tính cơ bản nhất. Nó thay thế giá trị của mỗi pixel bằng giá trị trung bình (mean) của các pixel nằm trong cửa sổ (kernel) lân cận. Hiệu quả trong việc làm mờ ảnh và giảm nhiễu Gaussian.",
        "formula": "Giá trị mới được tính bằng: $$g(x,y) = \\frac{1}{K} \\sum_{(s,t) \\in S} f(s,t)$$ Trong đó $K$ là số pixel trong cửa sổ $S$.",
        "purpose": "Làm mịn ảnh (blurring), giảm nhiễu ngẫu nhiên (random noise)."
    },
    "median_filter": {
        "name": "Lọc Trung vị (Median Filter)",
        "description": "Đây là bộ lọc phi tuyến tính. Nó thay thế giá trị của mỗi pixel bằng giá trị trung vị (median) của các pixel trong cửa sổ lân cận. Rất hiệu quả trong việc loại bỏ nhiễu Salt-and-Pepper (nhiễu hạt tiêu) mà ít làm mờ cạnh hơn lọc trung bình.",
        "formula": "Giá trị mới được tính bằng: $$g(x,y) = \\text{median} \\{ f(s,t) \\mid (s,t) \\in S \\}$$",
        "purpose": "Khử nhiễu phi tuyến tính, đặc biệt là nhiễu hạt tiêu."
    },
    "laplacian_sharpen": {
        "name": "Làm sắc nét bằng Bộ lọc Laplacian",
        "description": "Bộ lọc Laplacian là một bộ lọc đạo hàm bậc hai, được sử dụng để phát hiện cạnh và tăng cường chi tiết. Phép làm sắc nét được thực hiện bằng cách kết hợp ảnh gốc với kết quả của bộ lọc Laplacian.",
        "formula": "Làm sắc nét: $$g(x,y) = f(x,y) - \\nabla^2 f(x,y)$$ Trong đó $\\nabla^2 f(x,y)$ là kết quả của bộ lọc Laplacian.",
        "purpose": "Tăng cường chi tiết cạnh (sharpening), làm rõ các ranh giới trong ảnh."
    },
    "histogram_equalization": {
        "name": "Cân bằng Histogram",
        "description": "Thủ thuật cải thiện độ tương phản bằng cách 'kéo giãn' phạm vi cường độ pixel, làm cho các giá trị pixel sử dụng toàn bộ dải động có sẵn. Nó biến đổi phân bố histogram ban đầu thành một phân bố gần như đồng nhất.",
        "formula": "Giá trị cường độ mới $s_k$ được tính bằng: $$s_k = T(r_k) = (L-1) \\sum_{j=0}^{k} p_r(r_j)$$",
        "purpose": "Tăng cường độ tương phản ở các vùng ảnh tối hoặc sáng quá mức."
    },
    "gaussian_lowpass_filter": {
        "name": "Bộ lọc Thông thấp Gaussian (GLPF)",
        "description": "Bộ lọc làm mịn (smoothing) trong miền tần số. Nó giảm các thành phần tần số cao (chi tiết cạnh, nhiễu) bằng cách nhân với hàm Gaussian trong phổ Fourier. Không gây hiện tượng Ring.",
        "formula": "Hàm truyền bộ lọc: $$H(u,v)=e^{-D^{2}(u,v)/2D_{0}^{2}}$$ Trong đó $D(u,v)$ là khoảng cách từ trung tâm phổ, $D_0$ là tần số cắt.",
        "purpose": "Làm mịn ảnh, giảm nhiễu (noise reduction)."
    },
    "gaussian_highpass_filter": {
        "name": "Bộ lọc Thông cao Gaussian (GHPF)",
        "description": "Bộ lọc làm sắc nét (sharpening) trong miền tần số. Nó tăng cường các thành phần tần số cao (chi tiết cạnh) và giảm tần số thấp (vùng đồng nhất). Được xây dựng từ GLPF: $H_{hp} = 1 - H_{lp}$.",
        "formula": "$$H(u,v)=1-e^{-D^{2}(u,v)/2D_{0}^{2}}$$ Trong đó $D_0$ là tần số cắt.",
        "purpose": "Làm sắc nét, phát hiện cạnh (edge detection)."
    },
    "ideal_lowpass_filter": {
        "name": "Bộ lọc Thông thấp Lý tưởng (ILPF)",
        "description": "Bộ lọc làm mịn đơn giản nhất. Nó cắt hoàn toàn (cứng nhắc) tất cả các tần số nằm ngoài bán kính cắt $D_0$. Sự chuyển đổi đột ngột này tạo ra **Hiệu ứng Ring (Ringing Artifacts)** rõ rệt trong ảnh không gian.",
        "formula": "$$H(u,v)=\\begin{cases}1&\\text{nếu } D(u,v)\\le D_{0} \\\\ 0&\\text{nếu } D(u,v)>D_{0}\\end{cases}$$",
        "purpose": "Làm mịn ảnh (Chủ yếu dùng để minh họa hiệu ứng Ring)."
    },
    "ideal_highpass_filter": {
        "name": "Bộ lọc Thông cao Lý tưởng (IHPF)",
        "description": "Bộ lọc làm sắc nét cơ bản. Nó loại bỏ hoàn toàn các tần số thấp (vùng đồng nhất) và giữ lại các tần số cao. Cũng gây ra **Hiệu ứng Ring nghiêm trọng** trong ảnh kết quả.",
        "formula": "$$H(u,v)=\\begin{cases}0&\\text{nếu } D(u,v)\\le D_{0} \\\\ 1&\\text{nếu } D(u,v)>D_{0}\\end{cases}$$",
        "purpose": "Làm sắc nét cạnh (Chủ yếu dùng để minh họa hiệu ứng Ring khi làm sắc nét)."
    },

    "butterworth_lowpass_filter": {
        "name": "Bộ lọc Thông thấp Butterworth (BLPF)",
        "description": "Bộ lọc làm mịn có khả năng chuyển tiếp mượt mà giữa tần số thấp và tần số cao (kiểm soát bởi bậc $n$). Nó là một cải tiến lớn so với ILPF và **không tạo ra hiện tượng Ring**.",
        "formula": "$$H(u,v)=\\frac{1}{1+[D(u,v)/D_{0}]^{2n}}$$ Trong đó $n$ là bậc (order) của bộ lọc.",
        "purpose": "Làm mịn ảnh chất lượng cao mà không gây hiệu ứng Ring."
    },
    "butterworth_highpass_filter": {
        "name": "Bộ lọc Thông cao Butterworth (BHPF)",
        "description": "Bộ lọc làm sắc nét có chuyển tiếp mượt mà, giúp tăng cường cạnh mà **không gây hiện tượng Ring**. Độ dốc của sự chuyển tiếp được kiểm soát bởi bậc $n$.",
        "formula": "Hàm truyền bộ lọc: $$H(u,v)=\\frac{1}{1+[D_{0}/D(u,v)]^{2n}}$$ Trong đó $n$ là bậc (order) của bộ lọc.",
        "purpose": "Làm sắc nét cạnh và tăng cường chi tiết mà không tạo ra nhiễu Ring."
    },
    "contra_harmonic_mean": {
            "name": "Bộ lọc Trung bình Nghịch điều hòa",
            "description": "Bộ lọc tuyến tính, có khả năng chọn lọc loại bỏ nhiễu đơn cực. Nếu bậc Q > 0, nó loại bỏ nhiễu tiêu (pepper). Nếu Q < 0, nó loại bỏ nhiễu muối (salt). Nó được xem là một trong những bộ lọc mạnh mẽ nhất cho nhiễu xung.",
            "formula": "$$\\hat{f}(x,y)=\\frac{\\sum_{(s,t)\\in S_{xy}}g(s,t)^{Q+1}}{\\sum_{(s,t)\\in S_{xy}}g(s,t)^{Q}}$$ Trong đó, $S_{xy}$ là cửa sổ lọc.",
            "purpose": "Khử nhiễu muối (Salt) và nhiễu tiêu (Pepper) một cách có chọn lọc."
    },
    "adaptive_local_filter": {
        "name": "Bộ lọc Giảm nhiễu Cục bộ Thích nghi (Adaptive Local Noise)",
        "description": "Bộ lọc thông minh, tự động thay đổi mức độ lọc cho từng pixel dựa trên thống kê cục bộ (trung bình và phương sai cục bộ $\\sigma_L^2$) so với phương sai nhiễu đã biết ($\\sigma_{\\eta}^2$). Nó giữ lại chi tiết cạnh tốt hơn nhiều so với bộ lọc tuyến tính cố định.",
        "formula": "Giá trị ước lượng: $$\\hat{f}(x,y)=g(x,y)-\\frac{\\sigma_{\\eta}^{2}}{\\sigma_{L}^{2}}[g(x,y)-m_{L}]$$",
        "purpose": "Giảm nhiễu hiệu suất cao trong khi bảo toàn các chi tiết cạnh."
    },
    "inverse_filter": {
        "name": "Lọc Nghịch đảo (Inverse Filtering)",
        "description": "Thủ thuật phục hồi ảnh trong miền tần số. Nó được sử dụng để loại bỏ sự làm mờ (deblurring) gây ra bởi quá trình suy giảm tuyến tính, bất biến theo vị trí (ví dụ: chuyển động của camera). Đây là phương pháp khôi phục ảnh đơn giản nhất.",
        "formula": "Ước lượng ảnh gốc: $$\\hat{F}(u,v) = \\frac{G(u,v)}{H(u,v)}$$ Trong đó $G(u,v)$ là ảnh bị suy giảm, và $H(u,v)$ là hàm suy giảm ước lượng. (Cần giới hạn tần số để tránh khuếch đại nhiễu).",
        "purpose": "Khôi phục ảnh bị mờ do chuyển động (motion blur) hoặc hiệu ứng khí quyển bằng cách đảo ngược hàm suy giảm."
    },
    "otsu_segmentation": {
        "name": "Ngưỡng hóa Otsu (Otsu Thresholding)",
        "description": "Là một phương pháp ngưỡng hóa tự động, không cần tham số. Thuật toán Otsu tìm giá trị ngưỡng tối ưu để chia ảnh thành hai lớp (nền và đối tượng) bằng cách tối đa hóa phương sai giữa các lớp (inter-class variance).",
        "formula": "$$\\sigma^2_B(T) = w_0(\\mu_0 - \mu_T)^2 + w_1(\\mu_1 - \mu_T)^2$$ Tối đa hóa $\\sigma^2_B$ để tìm ngưỡng $T$ tối ưu.",
        "purpose": "Phân vùng ảnh nhị phân tự động, hiệu quả khi histogram có hai đỉnh rõ rệt."
    },
    "kmeans_segmentation": {
        "name": "K-Means Clustering cho Phân vùng",
        "description": "K-Means là thuật toán học máy không giám sát, được dùng để phân vùng ảnh bằng cách nhóm các pixel lại với nhau. Các pixel gần nhau trong không gian màu (RGB) và/hoặc không gian vị trí (XY) được coi là thuộc cùng một phân đoạn (segment).",
        "formula": "Tối thiểu hóa: $$J = \\sum_{k=1}^{K} \\sum_{i \\in S_k} ||x_i - \mu_k||^2$$ Trong đó $K$ là số nhóm (clusters), $x_i$ là pixel, và $\mu_k$ là trung tâm của nhóm $k$.",
        "purpose": "Phân đoạn ảnh dựa trên màu sắc hoặc kết cấu, giảm số lượng màu."
    },
    "mean_shift_segmentation": {
        "name": "Mean Shift Clustering cho Phân vùng",
        "description": "Mean Shift là thuật toán phân nhóm phi tham số (non-parametric). Nó tìm kiếm các chế độ (modes) hoặc các đỉnh mật độ trong phân phối dữ liệu. Phù hợp cho phân vùng ảnh vì nó không yêu cầu xác định trước số lượng nhóm (K) và giữ được ranh giới tự nhiên.",
        "formula": "Cập nhật dịch chuyển trung bình: $$m(x) = \\frac{\\sum_{i=1}^{n} K(x_i - x) x_i}{\\sum_{i=1}^{n} K(x_i - x)}$$ Tìm kiếm vector dịch chuyển để hội tụ về tâm khối (centroid).",
        "purpose": "Phân đoạn ảnh dựa trên mật độ dữ liệu màu, không cần xác định K trước."
    }
    
}