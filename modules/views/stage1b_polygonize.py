import streamlit as st
import os
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
from shapely.geometry import shape
import numpy as np

def render_stage1b():
    st.header("Giai đoạn 1b: Vector hóa Bản đồ (Polygonize)")
    st.caption("Chuyển đổi bản đồ Raster (.tif) thành Vector Polygon (.geojson) để làm ranh giới chính thức.")

    project_dir = st.session_state['project_dir']
    raster_path = os.path.join(project_dir, 'sugarcane_boundary.tif')
    sugarcane_boundary_path = os.path.join(project_dir, 'sugarcane_boundary.geojson')

    if os.path.exists(sugarcane_boundary_path):
        st.success("✅ Đã có Ranh giới Vector. Bạn có thể sang Giai đoạn 1c hoặc Polygonize lại.")
        col_next, col_del = st.columns([2, 1])
        with col_next:
            if st.button("Tiếp tục → Giai đoạn 1c (Tải Covariates) ➡️", type="primary", use_container_width=True):
                st.session_state['current_view'] = 'stage1c'
                st.rerun()
        with col_del:
            if st.button("🗑️ Xóa File Vector"):
                try:
                    os.remove(sugarcane_boundary_path)
                    st.rerun()
                except PermissionError:
                    st.error("🔒 QGIS đang mở file này. Hãy tắt nó!")
        st.markdown("---")
        
    if not os.path.exists(raster_path):
        st.info("⚠️ Không tìm thấy file Raster `sugarcane_boundary.tif` từ Giai đoạn 1a. Vui lòng tải một file Raster (.tif) bên dưới nếu có.")
    else:
        st.success(f"✅ Hệ thống đã nhận diện `sugarcane_boundary.tif` từ Giai đoạn 1a.")
        
    with st.expander("Tải lên Raster tùy chỉnh (Nếu muốn thay thế)", expanded=not os.path.exists(raster_path)):
        uploaded_tif = st.file_uploader("Upload File TIF", type=['tif', 'tiff'])
        if uploaded_tif is not None:
            if st.button("Lưu Raster"):
                with open(raster_path, "wb") as f:
                    f.write(uploaded_tif.getvalue())
                st.success("Đã nạp file TIF thành công.")
                st.rerun()

    st.markdown("---")
    
    tab_vectorize, tab_upload = st.tabs([
        "🔄 Phương án 1: Vector hóa từ AI Raster (Bước 1a)",
        "📂 Phương án 2: Tải lên ranh giới Vector CÓ SẴN"
    ])

    with tab_vectorize:
        if os.path.exists(raster_path):
            st.subheader("⚙️ Xử lý Polygonize")
            colA, colB = st.columns([1, 1])
            with colA:
                target_value = st.number_input("Giá trị Pixel đích từ Mức 1a (Biểu diễn vùng Mía):", min_value=0, value=255, step=1, key="target_val_input")
                min_area_ha = st.number_input("Bộ lọc Diện tích (Hecta):", min_value=0.0, value=0.0, step=0.1, key="min_area_input")
            with colB:
                st.caption("Thuật toán phân tách tất cả các pixel liền kề có giá trị này thành một đa giác (Polygon). Các pixel vùng Nền (0) sẽ bị loại bỏ.")
                st.caption("Diện tích tối thiểu giúp loại bỏ các nhiễu nhỏ (noise) trên bản đồ (VD: Lọc các mảnh ruộng dưới 0.1 ha).")
                
            force_utm = st.checkbox("Đồng bộ Gốc tọa độ Địa lý Địa phương (Local UTM)", value=True, help="Tự động bẻ cong bản đồ thẳng hàng với trục UTM trước khi cắt Vector. Điều này loại bỏ hoàn toàn hiện tượng 'Răng cưa viền méo' khi kết hợp với Giai đoạn 1c.")
                
            if st.button("🚀 Thực thi Polygonize", type="primary", use_container_width=True):
                with st.spinner("Đang dò quét các ranh giới ảnh sang dữ liệu Vector... Xin vui lòng chờ."):
                    try:
                        with rasterio.open(raster_path) as src:
                            if force_utm and src.crs.to_string() == "EPSG:4326":
                                # Tính toán UTM CRS tự động dựa trên tọa độ trung tâm
                                lon = (src.bounds.left + src.bounds.right) / 2
                                lat = (src.bounds.bottom + src.bounds.top) / 2
                                utm_zone = int((lon + 180) / 6) + 1
                                utm_crs = f"EPSG:326{utm_zone}" if lat >= 0 else f"EPSG:327{utm_zone}"
                                
                                st.info(f"🔄 Đang chuyển không gian Vector sang Local {utm_crs} để sửa lỗi Răng Cưa...")
                                
                                transform, width, height = calculate_default_transform(
                                    src.crs, utm_crs, src.width, src.height, *src.bounds)
                                
                                # Căn chỉnh Lưới (Sacred Grid) cho Vector y hệt Giai đoạn 1c
                                from rasterio.transform import from_bounds
                                # Đoán phân giải (10m hoặc 30m)
                                scale_proj = np.round(abs(transform[0]) / 10) * 10
                                if scale_proj == 0: scale_proj = 10
                                
                                b_minx = transform[2]
                                b_maxy = transform[5]
                                b_maxx = b_minx + transform[0] * width
                                b_miny = b_maxy + transform[4] * height
                                
                                sacred_minx = np.floor(b_minx / scale_proj) * scale_proj
                                sacred_miny = np.floor(b_miny / scale_proj) * scale_proj
                                sacred_maxx = np.ceil(b_maxx / scale_proj) * scale_proj
                                sacred_maxy = np.ceil(b_maxy / scale_proj) * scale_proj
                                
                                width = int((sacred_maxx - sacred_minx) / scale_proj)
                                height = int((sacred_maxy - sacred_miny) / scale_proj)
                                transform = from_bounds(sacred_minx, sacred_miny, sacred_maxx, sacred_maxy, width, height)
                                
                                image = np.zeros((height, width), dtype=src.dtypes[0])
                                reproject(
                                    source=rasterio.band(src, 1),
                                    destination=image,
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=transform,
                                    dst_crs=utm_crs,
                                    resampling=Resampling.nearest)
                                
                                target_crs = utm_crs
                                target_transform = transform
                            else:
                                image = src.read(1)
                                target_crs = src.crs
                                target_transform = src.transform

                            # mask filter out pixels that do not match the target value
                            mask = (image == target_value)
                            
                            results = (
                                {'properties': {'raster_val': v}, 'geometry': s}
                                for i, (s, v) 
                                in enumerate(shapes(image, mask=mask, transform=target_transform))
                            )
                            
                            geoms = list(results)
                            if len(geoms) == 0:
                                st.warning(f"❌ Không tìm thấy pixel nào với giá trị {target_value}!")
                            else:
                                gdf = gpd.GeoDataFrame.from_features(geoms)
                                gdf.set_crs(target_crs, inplace=True)
                                
                                # Clean up invalid geometries if any
                                gdf['geometry'] = gdf.geometry.buffer(0)
                                
                                # Filter minimum area
                                if min_area_ha > 0:
                                    areas_ha = gdf.to_crs("EPSG:3857").geometry.area / 10000
                                    gdf = gdf[areas_ha >= min_area_ha]
                                
                                # Không ép về EPSG:4326 nữa để bảo toàn trục tọa độ nếu đang ở hệ UTM
                                
                                if len(gdf) == 0:
                                    st.warning(f"❌ Không có polygon nào đạt chuẩn diện tích tối thiểu {min_area_ha} ha sau khi lọc!")
                                else:
                                    gdf.to_file(sugarcane_boundary_path, driver='GeoJSON')
                                    
                                    st.success(f"🎉 Rút trích thành công {len(gdf)} thửa ruộng (Polygons)!")
                                    st.info("Đã lưu ranh giới Mía thành `.geojson`. Nhấn Nút Tiếp tục phía trên để tiến tới Giai đoạn 1c.")
                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi hệ thống: {e}")

    with tab_upload:
        st.subheader("Tải lên Ranh giới ruộng mía")
        st.write("Nếu bạn đã có sẵn file Polygon ranh giới ruộng mía, hãy upload trực tiếp.")
        st.caption("Hỗ trợ: file .GeoJSON hoặc file .ZIP (chứa toàn bộ bộ Shapefile .shp, .shx, .dbf, .prj)")
        
        uploaded_boundary = st.file_uploader(
            "Chọn file ranh giới mía",
            type=['geojson', 'zip'],
            key="upload_sugarcane_boundary"
        )
        
        if uploaded_boundary is not None:
            try:
                import glob
                temp_ext = '.zip' if uploaded_boundary.name.endswith('.zip') else '.geojson'
                temp_name = os.path.join(project_dir, f"temp_boundary{temp_ext}")
                with open(temp_name, "wb") as f:
                    f.write(uploaded_boundary.getvalue())
                
                if temp_ext == '.zip':
                    import zipfile, shutil
                    extract_dir = os.path.join(project_dir, "temp_unzipped_boundary")
                    if os.path.exists(extract_dir):
                        shutil.rmtree(extract_dir)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(temp_name, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    shp_files = glob.glob(os.path.join(extract_dir, "**", "*.shp"), recursive=True)
                    if not shp_files:
                        st.error("Không tìm thấy file .shp bên trong file ZIP! Hãy kiểm tra lại.")
                        st.stop()
                    else:
                        gdf_boundary = gpd.read_file(shp_files[0])
                else:
                    gdf_boundary = gpd.read_file(temp_name)
                
                if gdf_boundary.crs is None:
                    gdf_boundary.set_crs("EPSG:4326", inplace=True)
                    
                gdf_boundary = gdf_boundary.to_crs("EPSG:4326")
                
                # Sử dụng EPSG:6933 (Equal Area) thay vì 3857 (Web Mercator) để tính diện tích chuẩn xác
                area_ha = gdf_boundary.to_crs("EPSG:6933").geometry.area.sum() / 10000
                st.write(f"📐 Phát hiện **{len(gdf_boundary)} polygon(s)**, tổng diện tích thực tế **{area_ha:,.1f} ha**")
                
                if st.button("✅ Lưu làm Vùng lõi của Dự án", type="primary", key="confirm_boundary"):
                    gdf_boundary.to_file(sugarcane_boundary_path, driver="GeoJSON")
                    st.success("Tuyệt vời! Hệ thống đã ghi nhận Ranh giới mía.")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Đã có lỗi khi giải mã file cấu hình: {e}")
