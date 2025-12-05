# 👁️ Computer Vision API Demo (Dự án Giữa Kỳ)

Dự án này là một ứng dụng web Demo toàn diện mô phỏng các kỹ thuật **Xử lý ảnh số (Digital Image Processing)** và **Thị giác máy tính (Computer Vision)**, được xây dựng bằng **Flask (Python)**, **OpenCV**, và **Scikit-learn**.

Mục tiêu là cung cấp một nền tảng demo tương tác (Front-end) để minh họa lý thuyết hàn lâm bằng mô phỏng thực tế trên ảnh của người dùng.

## 🚀 Tính năng Nổi bật & Giá trị Độc đáo

* **Cấu trúc Theo Chương trình học:** Ứng dụng được tổ chức thành 6 Tab logic, phản ánh các chương học thuật chính của môn học.
* **Trực quan hóa Phổ Tần số:** Các bộ lọc miền tần số (Tab 2, 3) hiển thị Phổ Biên độ ($|F(u,v)|$) và Mặt nạ bộ lọc ($H(u,v)$) để chứng minh nguyên lý lọc.
* **Tối ưu Hiệu suất ML:** Mean Shift và các thuật toán ML nặng khác được tối ưu bằng kỹ thuật **Lấy mẫu (Sampling)** để đảm bảo ứng dụng chạy nhanh và ổn định, tránh lỗi timeout.
* **Hỗ trợ Công thức MathJax:** Công thức toán học ($\LaTeX$) phức tạp được hiển thị chính xác trên Front-end.
* **Download Client-side:** Tải ảnh đã xử lý về máy người dùng mà không tốn dung lượng lưu trữ trên server.

---

## ⚙️ Cài đặt & Khởi động

1.  **Cài đặt Thư viện:** Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Khởi động Server:** Chạy ứng dụng Flask:
    ```bash
    python app.py
    ```

3.  **Truy cập:** Mở trình duyệt tại `http://127.0.0.1:5000/`

---

## 📋 Danh sách Thủ thuật (Phân loại theo Tabs)

Dự án triển khai đầy đủ các thủ thuật xử lý ảnh cơ bản đến nâng cao:

### TAB 1: 💡 Cải thiện Hình ảnh (Điểm, Hist & Không gian)

| Thủ thuật | Loại | Mục đích |
| :--- | :--- | :--- |
| **Biến đổi Âm bản, Logarit, Gamma** | Xử lý Điểm | Điều chỉnh độ sáng, tương phản. |
| **Cân bằng Histogram** | Histogram | Tăng cường độ tương phản tự động. |
| **Lọc Không gian** (Mean, Median, Laplacian) | Lọc | Làm mịn/Khử nhiễu (Median) và Làm sắc nét (Laplacian). |

### TAB 2: 🌊 Lọc trong Miền Tần số (Frequency Domain)

| Thủ thuật | Loại | Đặc điểm |
| :--- | :--- | :--- |
| **Gaussian (GLPF/GHPF)** | Thông Thấp/Cao | Chuyển tiếp mượt, không gây Hiệu ứng Ring. |
| **Ideal (ILPF/IHPF)** | Thông Thấp/Cao | Gây **Hiệu ứng Ring** rõ rệt (Minh họa vấn đề chuyển đổi cứng nhắc). |
| **Butterworth (BLPF/BHPF)** | Thông Thấp/Cao | Chuyển tiếp mượt, không Ring. |

### TAB 3: 🩹 Xử lý Nâng cao & Phục hồi

| Thủ thuật | Loại | Mục đích |
| :--- | :--- | :--- |
| **Bộ lọc Nghịch điều hòa** | Thống kê | Loại bỏ chọn lọc nhiễu Muối (Salt) hoặc Tiêu (Pepper) (dùng bậc Q). |
| **Bộ lọc Giảm nhiễu Thích nghi** | Thích nghi | Tự động điều chỉnh mức độ lọc dựa trên phương sai cục bộ. |
| **Lọc Nghịch đảo** | Phục hồi | Khôi phục ảnh bị mờ do suy giảm tuyến tính (deblurring). |

### TAB 4: 📊 Phân vùng Ảnh (Segmentation)

| Thủ thuật | Loại | Đặc điểm |
| :--- | :--- | :--- |
| **Ngưỡng hóa Otsu** | Truyền thống | Tự động tìm ngưỡng tối ưu ($T$) để nhị phân hóa. |
| **K-Means Clustering** | ML (Clustering) | Phân nhóm pixel dựa trên màu sắc (K cố định). |
| **Mean Shift Clustering** | ML (Clustering) | Phân nhóm theo mật độ, không cần xác định K trước (K tự động). |

### TAB 5: 📦 Nén Ảnh

| Thủ thuật | Endpoint | Loại | Mục đích |
| :--- | :--- | :--- | :--- |
| **Nén JPEG** | `/api/process/jpeg_compression` | Nén có tổn hao | Giảm dung lượng file, minh họa nhiễu khối (blockiness) khi Quality thấp. |

### TAB 6: 💠 Xử lý Hình thái học (Morphology)

| Thủ thuật | Endpoint | Loại | Mục đích |
| :--- | :--- | :--- | :--- |
| **Xói mòn (Erosion)** | `/api/process/morphology` | Cơ bản | Làm mỏng vật thể, loại bỏ các pixel nhiễu nhỏ. |
| **Giãn nở (Dilation)** | `/api/process/morphology` | Cơ bản | Làm dày vật thể, lấp đầy các lỗ hổng nhỏ. |
| **Khai mở (Opening)** | `/api/process/morphology` | Kết hợp | Làm mượt đường viền (Erosion sau đó Dilation). |
| **Đóng (Closing)** | `/api/process/morphology` | Kết hợp | Lấp đầy lỗ hổng và nối các khoảng trống hẹp (Dilation sau đó Erosion). |
