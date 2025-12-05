# Computer Vision API

Dự án này là một API mô phỏng các kỹ thuật xử lý ảnh số và thị giác máy tính, được triển khai bằng Flask (Python). Mục tiêu là cung cấp một nền tảng demo tương tác (Front-end) để minh họa các thuật toán cơ bản, nâng cao, và Machine Learning trong xử lý ảnh.

## 🚀 Tính năng Nổi bật & Giá trị Độc đáo

* **Cấu trúc Theo Chương trình học:** Ứng dụng được tổ chức thành 5 Tab logic, phản ánh các chương học thuật chính (Cải thiện, Lọc Tần số, Phục hồi, Phân đoạn, Nén).
* **Trực quan hóa Phổ Tần số:** Các bộ lọc miền tần số (Tab 2, 3) hiển thị **Phổ Biên độ ($|F(u,v)|$)** và **Mặt nạ bộ lọc ($H(u,v)$)** để chứng minh nguyên lý lọc.
* **Tối ưu Hiệu suất ML:** Mean Shift và các thuật toán ML nặng khác được tối ưu bằng kỹ thuật **Lấy mẫu (Sampling)** để đảm bảo ứng dụng chạy nhanh và ổn định.
* **Hỗ trợ Công thức MathJax:** Công thức toán học ($\LaTeX$) phức tạp được hiển thị chính xác trên Front-end.
* **Download Client-side:** Tải ảnh đã xử lý về máy người dùng mà không tốn dung lượng lưu trữ trên server.

## ⚙️ Yêu cầu Hệ thống và Cài đặt

1.  **Cài đặt Python:** Đảm bảo bạn đang sử dụng Python 3.x.
2.  **Cài đặt Thư viện:** Cài đặt các thư viện cần thiết bằng file `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

    (Nội dung cơ bản của requirements.txt: `Flask`, `numpy`, `opencv-python`, `scikit-learn`)

3.  **Khởi động Server:** Chạy ứng dụng Flask:

    ```bash
    python app.py
    ```

    Truy cập giao diện tại: `http://127.0.0.1:5000/`

---

## Danh sách Thủ thuật (Phân loại theo Tabs)

API được chia thành các phần chính, tương ứng với các lĩnh vực trong xử lý ảnh:

### TAB 1: Cải thiện Hình ảnh (Điểm, Histogram & Không gian)

Các kỹ thuật cơ bản cải thiện độ sáng, độ tương phản, và lọc cục bộ.

| Thủ thuật | Endpoint | Loại | Tham số |
| :--- | :--- | :--- | :--- |
| **Biến đổi Âm bản** | `/api/process/negative_image` | Điểm | - |
| **Biến đổi Logarit** | `/api/process/log_transform` | Điểm | `c` |
| **Biến đổi Luật Công suất (Gamma)** | `/api/process/power_law_transform` | Điểm | `c`, `gamma` |
| **Cân bằng Histogram** | `/api/process/histogram_equalization` | Histogram | - |
| **Lọc Miền Không gian (Chung)** | `/api/process/spatial_filter` | Lọc | `filter_type` (mean/median/laplacian\_sharpen), `kernel_size` |

### TAB 2: Lọc trong Miền Tần số (Frequency Domain)

Các bộ lọc phức tạp sử dụng Biến đổi Fourier để làm mịn (Lowpass) hoặc làm sắc nét (Highpass).

| Thủ thuật | Endpoint | Loại | Tham số Bắt buộc |
| :--- | :--- | :--- | :--- |
| **Gaussian Lowpass (GLPF)** | `/api/process/gaussian_lowpass_filter` | LPF | `D0` |
| **Ideal Lowpass (ILPF)** | `/api/process/ideal_lowpass_filter` | LPF | `D0` |
| **Butterworth Lowpass (BLPF)** | `/api/process/butterworth_lowpass_filter` | LPF | `D0`, `n` (Order) |
| **Gaussian Highpass (GHPF)** | `/api/process/gaussian_highpass_filter` | HPF | `D0` |
| **Ideal Highpass (IHPF)** | `/api/process/ideal_highpass_filter` | HPF | `D0` |
| **Butterworth Highpass (BHPF)** | `/api/process/butterworth_highpass_filter` | HPF | `D0`, `n` (Order) |

### TAB 3: Xử lý Nâng cao & Phục hồi (Restoration)

Các bộ lọc thống kê và phục hồi tiên tiến, được thiết kế để xử lý các mô hình nhiễu cụ thể.

| Thủ thuật | Endpoint | Loại | Tham số |
| :--- | :--- | :--- | :--- |
| **Bộ lọc Nghịch điều hòa** | `/api/process/contra_harmonic_mean` | Thống kê | `kernel_size`, `Q` (Bậc) |
| **Bộ lọc Giảm nhiễu Thích nghi** | `/api/process/adaptive_local_filter` | Thống kê | `kernel_size`, Vùng Ước lượng Nhiễu (x\_start, y\_start, width, height) |
| **Lọc Nghịch đảo** | `/api/process/inverse_filter` | Phục hồi | `modelK`, `cutoff_ratio` (Giới hạn tần số) |

### TAB 4: Phân vùng Ảnh (Segmentation)

Sử dụng các thuật toán truyền thống và Machine Learning để phân chia ảnh thành các vùng có ý nghĩa.

| Thủ thuật | Endpoint | Loại | Tham số |
| :--- | :--- | :--- | :--- |
| **Ngưỡng hóa Otsu** | `/api/process/otsu_segmentation` | Truyền thống | - |
| **Phân vùng ML (Chung)** | `/api/process/ml_segmentation` | ML (Clustering) | `model_type` (kmeans/mean\_shift), `n_clusters`, `bandwidth` |

### TAB 5: Nén Ảnh

| Thủ thuật | Endpoint | Loại | Mục đích |
| :--- | :--- | :--- | :--- |
| **Nén JPEG** | `/api/process/jpeg_compression` | Nén có tổn hao | Giảm dung lượng file, minh họa nhiễu khối (blockiness) khi Quality thấp. |

---

## 🛠️ Tính năng Kỹ thuật Chính

* **Xử lý Ảnh Đa Kênh:** Tất cả các bộ lọc làm mịn và xử lý điểm đều có khả năng xử lý ảnh màu bằng cách chuyển đổi sang không gian màu YUV/HSV và chỉ lọc kênh cường độ (Y/V).
* **Trực quan hóa Phổ:** Các bộ lọc miền tần số (Tab 2, 3) trả về Biểu đồ Phổ Tần số $|F(u,v)|$, Phổ đã Lọc $|G(u,v)|$, và Hàm truyền $H(u,v)$ dưới dạng hình ảnh, cho phép quan sát trực tiếp quá trình lọc.
* **Phục hồi Ảnh:** Triển khai các thuật toán nâng cao như Lọc Nghịch điều hòa (chống nhiễu Salt/Pepper) và Lọc Thích nghi (chống nhiễu Gaussian).

* **MathJax Support:** Công thức LaTeX được hiển thị đẹp mắt trên Front-end để minh họa lý thuyết.


