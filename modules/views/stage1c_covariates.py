import streamlit as st
import geopandas as gpd
import ee
import os
import glob
import json
import datetime
from modules.gee_fetcher import verify_ee_init, fetch_dem_slope, fetch_ndvi, fetch_chirps, fetch_twi
from modules.local_geodata import sync_raster_to_master
from modules.views.settings import load_system_config

@st.cache_data(show_spinner=False)
def get_vector_columns(file_path):
    import geopandas as gpd
    try:
        import fiona
        with fiona.open(file_path) as src:
            return list(src.schema['properties'].keys())
    except:
        try:
            gdf = gpd.read_file(file_path, rows=1)
            return list(gdf.columns)
        except:
            gdf = gpd.read_file(file_path)
            return list(gdf.columns)

def render_stage1():
    st.header("Giai đoạn 1c: Tải Biến môi trường (Covariates)")
    st.caption("Các đặc tính Vệ tinh sẽ tự động được gọt dũa (clip) vừa khít với Ranh giới Mía ở Bước 1b.")
    
    project_dir = st.session_state['project_dir']
    cov_dir = os.path.join(project_dir, 'covariates')
    os.makedirs(cov_dir, exist_ok=True)
    
    # --- Ranh giới Mía (Sống còn) ---
    sugarcane_boundary_path = os.path.join(project_dir, 'sugarcane_boundary.geojson')
    if not os.path.exists(sugarcane_boundary_path):
        st.error("❌ Chưa có Ranh giới Vector Mía! Hãy quay lại Giai đoạn 1b để Vector hóa bản đồ.")
        return
        
    # --- BỘ NHỚ LƯU TRỮ TRẠNG THÁI ---
    stage1_config_path = os.path.join(project_dir, 'stage1b_config.json')
    if os.path.exists(stage1_config_path):
        with open(stage1_config_path, 'r', encoding='utf-8') as f:
            saved_conf = json.load(f)
    else:
        saved_conf = {}
        
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("A. Tải ảnh từ Google Earth Engine")
        config = load_system_config()
        gee_id = config.get("gee_project_id", "")
        
        if not gee_id:
            st.error("Chưa lưu GEE Project ID trong Cài đặt Hệ thống! Vui lòng quay lại Dashboard -> Cài đặt.")
        else:
            if not verify_ee_init(project=gee_id):
                st.warning("⚠️ Không thể kết nối GEE bằng Project ID hiện tại.")
            else:
                st.success("✅ Đã xác thực GEE qua System Core.")
        
        st.info("ℹ️ Hệ thống đang TỰ ĐỘNG áp dụng ranh giới Mía Vector (Stage 1b) làm tọa độ duy nhất tải dữ liệu.")
        
        # Đọc tham số tọa độ từ boundary (Tối ưu hóa tránh treo RAM)
        gdf_boundary = gpd.read_file(sugarcane_boundary_path)
        minx, miny, maxx, maxy = gdf_boundary.to_crs("EPSG:4326").total_bounds
        lon = (minx + maxx) / 2
        lat = (miny + maxy) / 2
        utm_zone = int((lon + 180) / 6) + 1
        default_crs = f"EPSG:{32600 + utm_zone if lat >= 0 else 32700 + utm_zone}"
        st.caption(f"📍 Tự động nhận diện UTM Zone gốc {utm_zone}. Tọa độ GEE: **{default_crs}**")

        st.info("💡 **Gợi ý Kỹ thuật:** Nên chọn khoảng thời gian quét lùi khoảng 3 tháng tính từ ngày đi lấy mẫu đất thực địa.")
        
        def_start = saved_conf.get('start_date', (datetime.datetime.now() - datetime.timedelta(days=90)).strftime("%Y-%m-%d"))
        def_end = saved_conf.get('end_date', datetime.datetime.now().strftime("%Y-%m-%d"))
        
        start_date = st.date_input("Ngày thu vệ tinh (Bắt đầu)", datetime.datetime.strptime(def_start, "%Y-%m-%d"))
        end_date = st.date_input("Ngày thu vệ tinh (Kết thúc)", datetime.datetime.strptime(def_end, "%Y-%m-%d"))
        
        crs_input = st.text_input("Hệ tọa độ Tải về (System CRS)", value=saved_conf.get('crs', default_crs))
        default_scale_idx = 1 if saved_conf.get('scale', 30) == 30 else 0
        scale_project = st.selectbox("Resolution Đồng nhất - Pixel (m)", [10, 30], index=default_scale_idx)
        
        st.markdown("Chọn Ảnh vệ tinh:")
        get_ndvi = st.checkbox("NDVI", value=saved_conf.get('get_ndvi', True))
        get_dem = st.checkbox("DEM, Slope & TWI", value=saved_conf.get('get_dem', True))
        get_rain = st.checkbox("Lượng mưa CHIRPS", value=saved_conf.get('get_rain', True))
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn_gee = st.button("☁️ KÉO DỮ LIỆU VỆ TINH (GEE)", type="primary", use_container_width=True)

    with col2:
        st.subheader("B. Lớp Phân loại Cục bộ (Categorical)")
        st.write("Dùng để nhúng Tên Loại đất hoặc Vùng Trồng. Tự động nội suy cắt vừa khít theo Ranh giới Mía.")
        
        use_categorical = st.checkbox("Sử dụng tệp Phân loại (Tùy chọn)", value=saved_conf.get('use_categorical', False))
        
        cat_file_path = os.path.join(project_dir, 'temp_upload_cat.txt') # Lưu giữ đường dẫn gốc
        selected_col = None
        file_name_out = "Soil_Class.tif"
        
        if use_categorical:
            local_raster = st.file_uploader("Upload file Phân loại (.tif, .geojson, .zip)", type=['tif', 'tiff', 'geojson', 'zip'])
            file_name_out = st.text_input("Lưu tên file (Mặc định):", saved_conf.get('file_name_out', "Soil_Class.tif"))
            
            if local_raster is not None:
                is_vector = local_raster.name.endswith('.geojson') or local_raster.name.endswith('.zip')
                temp_ext = '.zip' if local_raster.name.endswith('.zip') else ('.geojson' if is_vector else '.tif')
                temp_path = os.path.join(project_dir, f'temp_cat_upload{temp_ext}')
                
                need_write = True
                if os.path.exists(temp_path) and os.path.getsize(temp_path) == local_raster.size:
                    need_write = False
                    
                if need_write:
                    with open(temp_path, "wb") as f: f.write(local_raster.getbuffer())
                    with open(cat_file_path, "w") as f: f.write(temp_path)  # Cache Path
                
                if is_vector:
                    st.info("💡 Lựa chọn Cột thuộc tính để nhúng:")
                    try:
                        read_path = f"zip://{temp_path}" if temp_ext == '.zip' else temp_path
                        col_list = get_vector_columns(read_path)
                        default_idx = col_list.index(saved_conf['selected_col']) if saved_conf.get('selected_col') in col_list else 0
                        selected_col = st.selectbox("Cột Thuộc tính Nhóm:", col_list, index=default_idx)
                    except Exception as e:
                        st.error(f"Lỗi phân tích tệp: {e}")
            
            elif os.path.exists(cat_file_path):
                with open(cat_file_path, "r") as f: temp_path = f.read()
                if os.path.exists(temp_path):
                    st.info(f"Đang chờ file Lịch sử: `{os.path.basename(temp_path)}`")
                    is_vector = temp_path.endswith('.geojson') or temp_path.endswith('.zip')
                    if is_vector:
                        read_path = f"zip://{temp_path}" if temp_path.endswith('.zip') else temp_path
                        col_list = get_vector_columns(read_path)
                        default_idx = col_list.index(saved_conf['selected_col']) if saved_conf.get('selected_col') in col_list else 0
                        selected_col = st.selectbox("Cột Thuộc tính Nhóm:", col_list, index=default_idx)
        else:
            st.caption("Hãy điền hết cấu hình cột Tải vệ tinh và nhấn Nút Bắt đầu.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        btn_cat = st.button("🗺️ ĐÚC LỚP PHÂN LOẠI (LOCAL)", type="primary", use_container_width=True)

    st.markdown("---")
    
    # --- KHỐI THỰC THI CHUNG ---
    if btn_gee or btn_cat:
        if not gee_id or not verify_ee_init(project=gee_id):
            st.error("❌ MÁY CHỦ BỊ NGẮT: Kết nối GEE phi mã chưa sẵn sàng!")
        else:
            with st.status("Gói gọn dữ liệu môi trường...", expanded=True) as status:
                st.write("🧹 0️⃣ Dọn dẹp cache Giai đoạn 2 để tránh rác...")
                import shutil
                for d in ['samples', 'models', 'outputs']:
                    del_dir = os.path.join(project_dir, d)
                    if os.path.exists(del_dir):
                        try: shutil.rmtree(del_dir)
                        except: pass

                for k in list(st.session_state.keys()):
                    if k.startswith('s2_') or k.startswith('s3_') or k.startswith('s4_'):
                        del st.session_state[k]

                st.write("1️⃣ Xây dựng Không gian Lưới Thiêng (Sacred Grid) và Tọa độ Khung...")
                gdf = gpd.read_file(sugarcane_boundary_path)
                
                # Tính Lưới Thiêng (Sacred Grid) trước tiên để đảm bảo GEE bao trùm toàn bộ
                gdf_utm = gdf.to_crs(crs_input)
                b_minx, b_miny, b_maxx, b_maxy = gdf_utm.total_bounds
                
                pad_m = max(300, scale_project * 10)
                
                import numpy as np
                sacred_minx = np.floor((b_minx - pad_m) / scale_project) * scale_project
                sacred_miny = np.floor((b_miny - pad_m) / scale_project) * scale_project
                sacred_maxx = np.ceil((b_maxx + pad_m) / scale_project) * scale_project
                sacred_maxy = np.ceil((b_maxy + pad_m) / scale_project) * scale_project
                sacred_width = int((sacred_maxx - sacred_minx) / scale_project)
                sacred_height = int((sacred_maxy - sacred_miny) / scale_project)
                
                if btn_gee:
                    from shapely.geometry import box
                    sacred_box_utm = gpd.GeoSeries([box(sacred_minx, sacred_miny, sacred_maxx, sacred_maxy)], crs=crs_input)
                    sacred_box_wgs = sacred_box_utm.to_crs("EPSG:4326").total_bounds
                    
                    minx_wgs, miny_wgs, maxx_wgs, maxy_wgs = sacred_box_wgs
                    pad_wgs = 0.001 
                    roi = ee.Geometry.Rectangle([minx_wgs - pad_wgs, miny_wgs - pad_wgs, maxx_wgs + pad_wgs, maxy_wgs + pad_wgs])
                    
                    if get_dem: 
                        st.write("2️⃣ Kéo bản đồ DEM, Slope & TWI...")
                        fetch_dem_slope(roi, cov_dir, scale=scale_project, crs=crs_input)
                        fetch_twi(roi, cov_dir, scale=scale_project, crs=crs_input)
                    if get_ndvi: 
                        st.write("3️⃣ Kéo Chuỗi NDVI...")
                        fetch_ndvi(roi, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), cov_dir, scale=scale_project, crs=crs_input)
                    if get_rain: 
                        st.write("4️⃣ Kéo CHIRPS Rainfall...")
                        fetch_chirps(roi, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), cov_dir, scale=scale_project, crs=crs_input)
                
                if btn_cat:
                    if use_categorical and os.path.exists(cat_file_path):
                        st.write("5️⃣ Đúc file Lớp Phân loại Cục bộ...")
                        with open(cat_file_path, "r") as f: temp_path = f.read()
                        if os.path.exists(temp_path):
                            base_files = glob.glob(os.path.join(cov_dir, '*.tif'))
                            if base_files:
                                master_path = base_files[0]
                                for bf in base_files:
                                    if "DEM" in bf: master_path = bf
                                    
                                out_path = os.path.join(cov_dir, file_name_out if file_name_out else 'Soil_Class.tif')
                                is_vector = temp_path.endswith('.geojson') or temp_path.endswith('.zip')
                                try:
                                    if is_vector and selected_col:
                                        from modules.local_geodata import rasterize_vector_to_master
                                        read_path = f"zip://{temp_path}" if temp_ext == '.zip' else temp_path
                                        rasterize_vector_to_master(read_path, selected_col, master_path, out_path)
                                    else:
                                        sync_raster_to_master(temp_path, master_path, out_path, is_categorical=True)
                                except Exception as e:
                                    st.error(f"Lỗi Categorical: {e}")
                            else:
                                st.error("❌ Không tìm thấy Raster Vệ tinh nào làm chuẩn (Master Raster). Bạn phải chạy tải Vệ tinh (GEE) ít nhất 1 lần trước khi đúc Lớp Phân loại!")
                    else:
                        st.write("5️⃣ Không dùng Mask Cục bộ.")
                    
                st.write("6️⃣ Gò toàn bộ Pixel gốc vào Lưới chuẩn cục bộ & Xén viền Laser...")
                from modules.local_geodata import mask_raster_with_vector, align_raster_to_sacred_grid
                from rasterio.transform import from_bounds
                
                sacred_transform = from_bounds(sacred_minx, sacred_miny, sacred_maxx, sacred_maxy, sacred_width, sacred_height)

                new_rasters = glob.glob(os.path.join(cov_dir, '*.tif'))
                for nr in new_rasters:
                    try:
                        if "temp" not in nr: 
                            is_cat = "Soil_Class" in nr
                            align_raster_to_sacred_grid(nr, crs_input, sacred_transform, sacred_width, sacred_height, is_categorical=is_cat)
                            # Cắt xén viền với lớp buffer bao quanh tối ưu cho Kriging (0.5 pixel)
                            # (Đủ để giữ 1 pixel viền thẳng và 2 pixel ở góc/viền chéo do all_touched=True, giảm nhiễu)
                            mask_raster_with_vector(nr, sugarcane_boundary_path, buffer_distance=scale_project * 0.5)
                    except Exception as e: 
                        st.error(f"Lỗi chỉnh Lưới {os.path.basename(nr)}: {e}")
                
                try:
                    st.write("7️⃣ Thống kê phương sai PCA tự động...")
                    from modules.sampling_opt import stack_rasters, fit_pca_and_save
                    final_rasters = glob.glob(os.path.join(cov_dir, '*.tif'))
                    if len(final_rasters) >= 2:
                        X_valid, X_cat_valid, coords_valid, r_meta, valid_mask, features, feature_names = stack_rasters(final_rasters)
                        if X_valid is not None and X_valid.shape[1] >= 2:
                            X_pca, pca_info = fit_pca_and_save(X_valid, valid_mask, features, r_meta, cov_dir, project_dir)
                        else: st.write("- Bỏ qua PCA do không đủ Band liên tục.")
                    else: st.write("- Bỏ qua PCA do không đủ Band.")
                except:
                    pass
                
                final_conf = {
                    'start_date': start_date.strftime("%Y-%m-%d"),
                    'end_date': end_date.strftime("%Y-%m-%d"),
                    'crs': crs_input,
                    'scale': scale_project,
                    'get_ndvi': get_ndvi,
                    'get_dem': get_dem,
                    'get_rain': get_rain,
                    'use_categorical': use_categorical,
                    'file_name_out': file_name_out,
                }
                if selected_col is not None: final_conf['selected_col'] = selected_col
                with open(stage1_config_path, 'w', encoding='utf-8') as f:
                    json.dump(final_conf, f, indent=4)
                    
                status.update(label="Toàn bộ quy trình hoàn tất!", state="complete", expanded=False)

    st.markdown("---")
    if glob.glob(os.path.join(cov_dir, '*.tif')):
        st.success("🎉 **GIAI ĐOẠN 1C ĐÃ HOÀN TẤT** - Toàn bộ Biến môi trường TẠI Ranh giới Mía đã sẵn sàng.")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("Bước tiếp theo: GIAI ĐOẠN 2 (Thiết kế Lấy Mẫu) ➡️", use_container_width=True, type="primary"):
                st.session_state['current_view'] = 'stage2'
                st.rerun()
