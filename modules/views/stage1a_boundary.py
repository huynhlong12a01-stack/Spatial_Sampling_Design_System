import streamlit as st
import geopandas as gpd
import ee
import os
import glob
import numpy as np
from modules.views.settings import load_system_config

def render_stage1a():
    st.header("Giai đoạn 1a: Khởi tạo Ranh giới Ruộng Mía")
    st.caption("Bước cốt lõi: Xác định vùng trồng mía để Hệ thống biết chính xác khu vực cần rải điểm và tải dữ liệu biến môi trường.")
    
    project_dir = st.session_state['project_dir']
    sugarcane_boundary_tif = os.path.join(project_dir, 'sugarcane_boundary.tif')
    sugarcane_boundary_geojson = os.path.join(project_dir, 'sugarcane_boundary.geojson')
    
    # === KIỂM TRA TRẠNG THÁI HIỆN TẠI ===
    if os.path.exists(sugarcane_boundary_tif):
        st.success(f"✅ Đã tìm thấy File Raster (.tif): {os.path.basename(sugarcane_boundary_tif)}")
        
        if st.button("📂 Mở Thư Mục Dự Án (Nơi chứa file TIF)", type="primary", use_container_width=True):
            os.startfile(project_dir)
            
        st.info("Bản đồ TIF đã sẵn sàng. Hãy bấm nút Tiếp tục bên dưới để mở Giai đoạn 1b và chuyển Raster thành Vector (Polygonize).")
        
        col_del, col_next = st.columns([1, 1])
        with col_del:
            if st.button("🗑️ Xóa ranh giới (Làm lại Bước 1a)", type="secondary"):
                import shutil
                try:
                    if os.path.exists(sugarcane_boundary_tif): os.remove(sugarcane_boundary_tif)
                    ai_rasters_dir = os.path.join(project_dir, 'ai_rasters_output')
                    if os.path.exists(ai_rasters_dir): shutil.rmtree(ai_rasters_dir, ignore_errors=True)
                    for temp_file in glob.glob(os.path.join(project_dir, "temp_*")):
                        if os.path.isfile(temp_file): os.remove(temp_file)
                        elif os.path.isdir(temp_file): shutil.rmtree(temp_file, ignore_errors=True)
                except Exception as e: st.error(f"Lỗi khi xóa dữ liệu: {e}")
                st.rerun()
        with col_next:
            if st.button("Tiếp tục → Giai đoạn 1b (Vector hóa) ➡️", type="primary", use_container_width=True):
                st.session_state['current_view'] = 'stage1b'
                st.rerun()
        return

    st.markdown("---")
    st.info("💡 **LƯU Ý:** Nếu bạn **đã có sẵn** file Ranh giới dạng Vector (GeoJSON/Shapefile), bạn **KHÔNG CẦN** thực hiện bước giải đoán AI này. Hãy kéo xuống dưới và ấn nút **'Tiếp tục → Giai đoạn 1b'** để tải file Vector của bạn lên.")
    
    st.subheader("Sử dụng AI & Vệ tinh để tự động Khoanh vùng Mía")
    st.write("Cung cấp **Ranh giới khu vực rộng** (ROI như xã/huyện) và **vài tọa độ thực địa đã biết là mía**. Hệ thống sẽ tải lịch sử sinh trưởng từ Sentinel-2 và chạy Random Forest để tìm tất cả các lô mía tương đồng.")
    config = load_system_config()
    gee_id = config.get("gee_project_id", "")
    if not gee_id:
        st.error("Chưa cấu hình tài khoản Google Earth Engine (GEE Project ID). Vào phần Cài đặt Hệ thống.")
        return
        
    col_roi, col_gt = st.columns(2)
    
    with col_roi:
        st.markdown("#### 1️⃣ Ranh giới Khu vực lớn (ROI)")
        roi_file = st.file_uploader(
            "Upload Khu vực tìm kiếm (GeoJSON/ZIP)",
            type=['geojson', 'zip'],
            key="upload_roi"
        )
        
    with col_gt:
        st.markdown("#### 2️⃣ Điểm kiểm chứng (Ground Truth)")
        gt_file = st.file_uploader(
            "Upload tập Điểm tọa độ Mía (CSV/GeoJSON/ZIP)",
            type=['csv', 'geojson', 'zip'],
            key="upload_ground_truth"
        )
        
        selected_lon_col, selected_lat_col = None, None
        
        if gt_file and gt_file.name.lower().endswith('.csv'):
            import pandas as pd
            try:
                temp_df = pd.read_csv(gt_file)
                cols = list(temp_df.columns)
                def_lon = next((i for i, c in enumerate(cols) if c.lower() in ['lon', 'longitude', 'lng', 'x', 'long']), 0)
                def_lat = next((i for i, c in enumerate(cols) if c.lower() in ['lat', 'latitude', 'y']), min(1, len(cols)-1))
                
                selected_lon_col = st.selectbox("Chọn Cột Kinh độ (X/Lon):", cols, index=def_lon)
                selected_lat_col = st.selectbox("Chọn Cột Vĩ độ (Y/Lat):", cols, index=def_lat)
                st.caption("✨ CRS sẽ được hệ thống *tự động nhận diện* (EPSG:4326) dựa trên tọa độ.")
                gt_file.seek(0)
            except Exception as e:
                st.error(f"Lỗi đọc file CSV: {e}")
        
    alpha_year = st.selectbox("Năm thu hoạch trọng tâm:", [2025, 2024, 2023, 2022], index=0)
    
    res_mapping = {
        "30m (Cân bằng & Nhanh - Khuyên dùng)": 30,
        "10m (Chi tiết cao nhưng Rất Nặng)": 10
    }
    res_choice = st.radio("Độ phân giải Không gian (Grid Scale):", list(res_mapping.keys()), index=0)
    selected_res = res_mapping[res_choice]
    
    valid_ready = True
    gdf_roi = None
    gdf_gt = None
    
    if roi_file:
        try:
            gdf_roi = _load_vector(roi_file, project_dir, "temp_roi")
            st.success(f"✅ Đọc ROI thành công! ({len(gdf_roi)} vùng)")
        except Exception as e:
            st.error(f"Lỗi ROI: {e}"); valid_ready = False
    else:
        valid_ready = False
        
    if gt_file:
        try:
            gdf_gt = _load_ground_truth(gt_file, project_dir, lon_col=selected_lon_col, lat_col=selected_lat_col)
            if gdf_gt is not None:
                st.success(f"✅ Đọc GT thành công! ({len(gdf_gt)} tọa độ)")
            else: valid_ready = False
        except Exception as e:
            st.error(f"Lỗi GT: {e}"); valid_ready = False
    else:
        valid_ready = False
        
    if valid_ready:
        if st.button("🚀 Kích hoạt Tải Dữ liệu & Giải đoán AI", type="primary", use_container_width=True):
            _run_interpretation(project_dir, gdf_roi, gdf_gt, alpha_year, selected_res, gee_id, sugarcane_boundary_tif)
                
    st.markdown("---")
    if st.button("Tiếp tục → Giai đoạn 1b (Xử lý Vector) ➡️", type="primary", use_container_width=True):
        st.session_state['current_view'] = 'stage1b'
        st.rerun()

# =============================================
# CÁC HÀM XỬ LÝ (Helpers)
# =============================================

def _load_vector(uploaded_file, project_dir, prefix):
    """Tiện ích tải Vector (GeoJSON/Zip) dùng chung."""
    name = uploaded_file.name.lower()
    temp_ext = '.zip' if name.endswith('.zip') else '.geojson'
    temp_path = os.path.join(project_dir, f"{prefix}{temp_ext}")
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
        
    if temp_ext == '.zip':
        import zipfile, glob, shutil
        extract_dir = os.path.join(project_dir, f"{prefix}_unzip")
        if os.path.exists(extract_dir): shutil.rmtree(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(temp_path, 'r') as z:
            z.extractall(extract_dir)
        shp_files = glob.glob(os.path.join(extract_dir, "**", "*.shp"), recursive=True)
        if shp_files:
            gdf = gpd.read_file(shp_files[0])
            if gdf.crs is None: gdf.set_crs("EPSG:4326", inplace=True)
            return gdf.to_crs("EPSG:4326")
        raise ValueError("Không tìm thấy tệp .shp trong thư mục ZIP.")
    else:
        gdf = gpd.read_file(temp_path)
        if gdf.crs is None: gdf.set_crs("EPSG:4326", inplace=True)
        return gdf.to_crs("EPSG:4326")

def _load_ground_truth(uploaded_file, project_dir, lon_col=None, lat_col=None):
    """Xử lý ranh giới Điểm. CRS được tự động phân tích không cần User nhập tay."""
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        import pandas as pd
        import numpy as np
        from shapely.geometry import Point
        temp_path = os.path.join(project_dir, "temp_gt.csv")
        with open(temp_path, "wb") as f: f.write(uploaded_file.getvalue())
        
        df = pd.read_csv(temp_path)
        use_lon = lon_col if lon_col else next((c for c in df.columns if c.lower() in ['lon', 'longitude', 'lng', 'x', 'long']), None)
        use_lat = lat_col if lat_col else next((c for c in df.columns if c.lower() in ['lat', 'latitude', 'y']), None)
        
        if not use_lon or not use_lat:
            raise ValueError(f"Không tìm được cột tọa độ trong: {list(df.columns)}")
            
        # TỰ ĐỘNG CHUẨN ĐOÁN CRS THEO KINEMATICS
        max_x = np.nanmax(np.abs(df[use_lon]))
        max_y = np.nanmax(np.abs(df[use_lat]))
        
        if max_x <= 180 and max_y <= 90:
            auto_crs = "EPSG:4326"
        else:
            # File CSV đơn thuần chỉ lưu số, không nhúng metadata CRS. 
            # Nếu tọa độ lớn hơn 180 (hệ Projected) mà không có header, máy sẽ mù tịt.
            raise ValueError("Phát hiện tọa độ quá quy chuẩn WGS84 (X > 180). File CSV không bao hàm mã CRS cục bộ. Vui lòng sử dụng phần mềm GIS (QGIS/ArcGIS) xuất file này ra dưới định dạng GeoJSON hoặc WGS84 trước khi Upload!")
            
        geometry = [Point(xy) for xy in zip(df[use_lon], df[use_lat])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=auto_crs)
        return gdf.to_crs("EPSG:4326")
    else:
        return _load_vector(uploaded_file, project_dir, "temp_gt")

def _run_interpretation(project_dir, gdf_roi, gdf_gt, target_year, resolution, gee_id, out_boundary_path):
    from modules.cloud_rf_engine import run_gee_pipeline
    from modules.gee_fetcher import verify_ee_init
    
    if not verify_ee_init(project=gee_id):
        st.error("Không kết nối được GEE!")
        return
        
    with st.status("Quy trình Giải đoán Đám mây (Google Earth Engine)...", expanded=True) as status:
        try:
            final_tif_path = run_gee_pipeline(gdf_roi, gdf_gt, target_year, out_boundary_path, resolution, status)
            if final_tif_path is not None and os.path.exists(final_tif_path):
                status.update(label=f"🌟 HOÀN THÀNH! Tải dữ liệu thành công -> Tệp TIF được lưu tại Dự án.", state="complete")
                st.success(f"🌟 Tiến trình AI kết thúc thành công! Raster đã lưu ở {final_tif_path}")
                st.rerun()
            else:
                status.update(label="❌ RF Không có bất kỳ ô mía nào xuất hiện.", state="error")
                st.error("❌ Random Forest dự báo Không có bất kỳ ô mía nào xuất hiện hoặc bị lỗi trích xuất File.")
        except Exception as e:
            import traceback
            status.update(label=f"❌ Lỗi GEE Pipeline: {e}", state="error")
            st.code(traceback.format_exc())
