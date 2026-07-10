import folium
import geopandas as gpd
import pandas as pd
import os
import glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import matplotlib.cm as cm

def get_wgs84_image(raster_path, max_size=1200):
    """
    Kỹ thuật Anti-Lag: 
    Đọc một ảnh Vệ tinh/Raster, Reproject sang hệ WGS84,
    Ép giảm độ phân giải xuống max_size px để trình duyệt web hiển thị mượt mà.
    Phủ màu tự động tùy vào đặc tính ảnh.
    Trở về Numpy RGBA matrix và tọa độ [ [South, West], [North, East] ] của Folium.
    """
    with rasterio.open(raster_path) as src:
        # Tỉ lệ khung hình
        aspect = src.width / src.height
        if src.width > src.height:
            dst_width = max_size
            dst_height = int(max_size / aspect)
        else:
            dst_height = max_size
            dst_width = int(max_size * aspect)
            
        dst_crs = 'EPSG:4326'
        
        # Cấu hình phép nén đổi Tọa độ
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, 
            dst_width=dst_width, dst_height=dst_height
        )
        
        out_image = np.zeros((1, height, width), dtype=np.float32)
        
        # Xóa cứng Resampling Bilinear cho các lớp rời rạc
        file_lower = os.path.basename(raster_path).lower()
        is_categorical = any(kw in file_lower for kw in ["class", "cluster", "mask", "alphaearth", "sugarcane"])
        resampling_method = Resampling.nearest if is_categorical else Resampling.bilinear
        
        reproject(
            source=rasterio.band(src, 1),
            destination=out_image,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling_method
        )
        
        band = out_image[0]
        
        # Lấy nhãn Vô Cực (NoData)
        nodata = src.nodata
        if nodata is not None:
            mask = (np.isclose(band, nodata)) | (np.isnan(band))
        else:
            mask = np.isnan(band)
            
        # Tìm Khung viền Bounding Box cho Folium (South, West -> North, East)
        w, s, e, n = rasterio.transform.array_bounds(height, width, transform)
        bounds = [[s, w], [n, e]]
        
        valid_data = band[~mask]
        if len(valid_data) == 0:
            return None, bounds
            
        # Nhúng Tông Màu (Colormap) tự động dựa vào tên File
        # Xử lý dải Stretch tùy thuộc vào dữ liệu Continuous hay Categorical
        if "alphaearth" in file_lower or "cropcluster" in file_lower or "mask" in file_lower:
            vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
        else:
            vmin, vmax = np.percentile(valid_data, 2), np.percentile(valid_data, 98)
            
        if vmax <= vmin: vmax = vmin + 0.001 # Tránh chia cho 0
        norm_band = np.clip((band - vmin) / (vmax - vmin), 0, 1)
        
        if "ndvi" in file_lower: 
            cmap = cm.get_cmap('RdYlGn')
        elif "dem" in file_lower or "slope" in file_lower: 
            cmap = cm.get_cmap('terrain')
        elif "chirps" in file_lower or "rain" in file_lower: 
            cmap = cm.get_cmap('Blues')
        elif "uncertainty" in file_lower:
            cmap = cm.get_cmap('YlOrRd')
        elif "residual" in file_lower:
            cmap = cm.get_cmap('RdBu_r') 
        elif "alphaearth" in file_lower or "cropcluster" in file_lower:
            cmap = cm.get_cmap('tab10')
        elif "sugarcane" in file_lower or ("mask" in file_lower and "alphaearth" not in file_lower):
            from matplotlib.colors import LinearSegmentedColormap
            # Xanh Chuối Dạ Quang - chỉ pixel hợp lệ (=1.0) mới sáng, pixel NaN sẽ trong suốt
            cmap = LinearSegmentedColormap.from_list('sugarcane', ['#39FF14', '#39FF14'], N=2)
        else:
            cmap = cm.get_cmap('viridis')
            
        rgba_img = cmap(norm_band) # Shape là: (Height, Width, 4)
        
        # Khoét rỗng Alpha (Trong suốt) cho các điểm NoData
        rgba_img[mask, 3] = 0.0
        
        return rgba_img, bounds

def build_project_map(project_dir):
    """Xây dựng và Lắp ghép Toàn bộ Layer của Dự án lên 1 mảng Giao diện Bản đồ Web"""
    
    # Khởi tạo bản đồ Ống nhòm (Trống, nền Mặc định)
    m = folium.Map(location=[16.0, 106.0], zoom_start=5, tiles=None)
    
    # 1. Thêm nền Bản đồ Google Phân bón hỗn hợp (Satellite + Map)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='🛰️ Bản đồ Vệ tinh Vô hướng (Nền chính)',
        overlay=False,
        control=False
    ).add_to(m)
    
    map_center = None
    
    # 2. XẾP CHỒNG ĐƯỜNG KẼ TỌA ĐỘ RANH GIỚI
    boundary_path = os.path.join(project_dir, 'sugarcane_boundary.geojson')
    fallback_path = os.path.join(project_dir, 'roi.geojson')
    
    # 2a. Vẽ khung ROI tổng thể (nếu có)
    if os.path.exists(fallback_path):
        try:
            gdf_roi = gpd.read_file(fallback_path)
            if gdf_roi.crs != 'EPSG:4326':
                gdf_roi = gdf_roi.to_crs('EPSG:4326')
            
            # Khóa ống kính màn hình Zoom thẳng vào giữa Khung
            bounds = gdf_roi.total_bounds
            map_center = [(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2]
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
            
            folium.GeoJson(
                gdf_roi,
                name='🔲 Ranh giới Khu vực (ROI)',
                style_function=lambda x: {'fillColor': '#00000000', 'color': '#FF3333', 'weight': 3, 'dashArray': '5, 5'}
            ).add_to(m)
        except Exception as e:
            print("Lỗi Map ROI Polygon:", e)

    # 2b. Vẽ Phủ lấp màu Xanh lá cho Lô Mía (nếu có)
    if os.path.exists(boundary_path):
        try:
            gdf_sg = gpd.read_file(boundary_path)
            if gdf_sg.crs != 'EPSG:4326':
                gdf_sg = gdf_sg.to_crs('EPSG:4326')
            
            if map_center is None:
                bounds = gdf_sg.total_bounds
                map_center = [(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2]
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                
            folium.GeoJson(
                gdf_sg,
                name='🌾 Lô Mía (Sugarcane)',
                style_function=lambda x: {
                    'fillColor': '#00FF00',  # Xanh lá sáng chói
                    'fillOpacity': 0.65,     # Độ trong suốt để nhìn thấy nền vệ tinh
                    'color': '#00CC00',      # Viền xanh đậm hơn
                    'weight': 1.5            # Độ dày viền nhỏ
                }
            ).add_to(m)
        except Exception as e:
            print("Lỗi Map Sugarcane Polygon:", e)
            
    # 3. ÉP TRUYỀN HÌNH ẢNH RASTER Băng Thông Thấp
    cov_dir = os.path.join(project_dir, 'covariates')
    tifs = glob.glob(os.path.join(cov_dir, '*.tif'))
    for t in tifs:
        name = os.path.basename(t)
        # Chỉ những File hoàn thiện (bỏ file temp)
        if "temp" in name.lower() or "soil_class" in name.lower() or "sugarcane_mask" in name.lower() or "alphaearth" in name.lower() or "ndvi_multitemporal" in name.lower():
            continue 
            
        try:
            rgba_img, bnds = get_wgs84_image(t, max_size=1200) # Đã bỏ giới hạn nén để giữ độ phân giải thật
            if rgba_img is not None:
                if map_center is None:
                    map_center = [(bnds[0][0]+bnds[1][0])/2, (bnds[0][1]+bnds[1][1])/2]
                    m.fit_bounds(bnds)
                    
                folium.raster_layers.ImageOverlay(
                    image=rgba_img,
                    bounds=bnds,
                    opacity=0.85,
                    name=f"🗺️ Lớp môi trường: {name.split('.')[0]}",
                    show=False # Mặc định ẩn để trống màn hình
                ).add_to(m)
        except Exception as e:
            print(f"Lỗi Render TIF {name}:", e)
            
    # 3.5 HIỂN THỊ CROP CLUSTER KMEANS LAYER (Bản đồ Phân Cụm AI)
    cluster_files = glob.glob(os.path.join(project_dir, 'CropCluster_KMeans_Y*.tif')) + glob.glob(os.path.join(project_dir, 'AlphaEarth_KMeans_Y*.tif'))
    for ae_f in cluster_files:
        try:
            rgba_img, bnds = get_wgs84_image(ae_f, max_size=1200)
            if rgba_img is not None:
                folium.raster_layers.ImageOverlay(
                    image=rgba_img,
                    bounds=bnds,
                    opacity=0.75,
                    name=f"🎨 Phân cụm AI: {os.path.basename(ae_f).split('.')[0]}",
                    show=False
                ).add_to(m)
        except Exception as e:
            print(f"Lỗi Render KMeans:", e)
    
    # (Sugarcane_Mask.tif render block removed)
            
    # 4. CHIA CỌC GẮN LÊN BẢN ĐỒ (SAMPLING POINTS X/Y)
    spl_csv = os.path.join(project_dir, 'samples', 'optimal_samples.csv')
    if os.path.exists(spl_csv):
        try:
            df = pd.read_csv(spl_csv)
            if 'Longitude' in df.columns and 'Latitude' in df.columns:
                fg_samples = folium.FeatureGroup(name="📍 Tọa độ Lấy mẫu Toán học", show=True)
                for idx, row in df.iterrows():
                    lon, lat = row['Longitude'], row['Latitude']
                    if pd.notna(lon) and pd.notna(lat):
                        pid = row.get('Point_ID', f"P{idx+1}")
                        
                        html_popup = f"""
                        <div style="font-family: Arial; font-size: 14px;">
                            <b>{pid}</b><br>
                            Kinh độ (X): {lon:.6f}<br>
                            Vĩ độ (Y): {lat:.6f}
                        </div>
                        """
                        
                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(html_popup, max_width=250),
                            tooltip=f"Ghim: {pid}",
                            icon=folium.Icon(color='red', icon='leaf')
                        ).add_to(fg_samples)
                        
                fg_samples.add_to(m)
        except Exception as e:
            print("Lỗi Render Điểm Lý Thuyết:", e)
            
    # 4.5. ĐIỂM LẤY MẪU THỰC TẾ (REAL SAMPLES TỪ FILE LAB)
    real_csv = os.path.join(project_dir, 'samples', 'real_samples.csv')
    if os.path.exists(real_csv):
        try:
            df_real = pd.read_csv(real_csv)
            if 'Longitude' in df_real.columns and 'Latitude' in df_real.columns:
                fg_real = folium.FeatureGroup(name="🟢 Mẫu Thực địa (Đã phân tích)", show=True)
                for idx, row in df_real.iterrows():
                    lon, lat = row['Longitude'], row['Latitude']
                    if pd.notna(lon) and pd.notna(lat):
                        pid_real = f"Real-{idx+1}"
                        # Thử lấy thêm thông số để hiển thị nếu có
                        extra_info = "<br>".join([f"{c}: {row[c]}" for c in df_real.columns if c not in ['Longitude', 'Latitude', 'Proj_X', 'Proj_Y']][:3])
                        
                        html_popup_real = f"""
                        <div style="font-family: Arial; font-size: 14px;">
                            <b>{pid_real}</b><br>
                            Kinh độ (X): {lon:.6f}<br>
                            Vĩ độ (Y): {lat:.6f}<br>
                            -----<br>
                            <span style="color: green; font-size: 12px;">{extra_info}...</span>
                        </div>
                        """
                        
                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(html_popup_real, max_width=250),
                            tooltip=f"Ghim Thực địa: {pid_real}",
                            icon=folium.Icon(color='green', icon='ok-sign')
                        ).add_to(fg_real)
                        
                fg_real.add_to(m)
        except Exception as e:
            print("Lỗi Render Điểm Thực Tế:", e)
            
    # 5. HIỂN THỊ KẾT QUẢ CÁC BẢN ĐỒ KRIGING (NẾU CÓ)
    out_dir = os.path.join(project_dir, 'outputs')
    out_tifs = glob.glob(os.path.join(out_dir, '*.tif'))
    for kt in out_tifs:
        name = os.path.basename(kt).split('.')[0]
        # Bỏ qua nếu có file không chuẩn
        if "Report_" in name: continue
        
        # Định tuyến hiển thị
        if name.startswith("RK_Map_"):
            layer_name = f"💎 {name}"
            is_show = True
        elif name.startswith("Uncertainty_"):
            layer_name = f"⚠️ Cảnh báo Lỗi: {name}"
            is_show = False
        elif name.startswith("Trend_"):
            layer_name = f"📈 Gốc Xu hướng (ML Trend): {name}"
            is_show = False
        elif name.startswith("Residual_"):
            layer_name = f"📉 Bù trừ Không gian (Residuals): {name}"
            is_show = False
        else:
            continue
            
        try:
            rgba_img, bnds = get_wgs84_image(kt, max_size=1600) # Bản đồ đầu ra siêu nét 1600px
            if rgba_img is not None:
                folium.raster_layers.ImageOverlay(
                    image=rgba_img,
                    bounds=bnds,
                    opacity=0.85,
                    name=layer_name,
                    show=is_show
                ).add_to(m)
        except Exception as e:
            print(f"Lỗi Render TIF {name}:", e)
            
    if map_center is None:
        m.location = [14.0583, 108.2772] 
        
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m
