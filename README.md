# Regression Kriging App

Ứng dụng web tĩnh để nội suy Regression Kriging từ điểm đo lat/lon/value, ROI GeoJSON và variogram phần dư.

## Cách chạy

Mở file index.html bằng trình duyệt.

## Dữ liệu đầu vào

- CSV điểm đo: cần có cột vĩ độ, kinh độ và chỉ tiêu phân tích.
- ROI: GeoJSON Polygon hoặc MultiPolygon, hệ tọa độ WGS84, thứ tự tọa độ [lon, lat].
- Variogram: hỗ trợ Spherical, Exponential, Gaussian; có thể nhập thủ công nugget, partial sill, range hoặc bấm Fit variogram.

## Kết quả

- Bản đồ nội suy trên lưới nằm trong ROI.
- Biểu đồ variogram thực nghiệm của phần dư hồi quy.
- Xuất kết quả dạng CSV hoặc GeoJSON điểm tâm ô lưới.
