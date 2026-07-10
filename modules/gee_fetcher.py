import ee
import geemap
import os
import streamlit as st

@st.cache_resource(show_spinner=False)
def verify_ee_init(project=None):
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        return False

def _estimate_tile_grid(roi, scale, n_bands=1, max_bytes=25_000_000):
    """Tính số ô lưới cần chia để mỗi tile < 25MB (GEE limit an toàn)."""
    import math
    bounds = roi.bounds().getInfo()['coordinates'][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    minx, maxx = min(lons), max(lons)
    miny, maxy = min(lats), max(lats)
    
    # Ước lượng kích thước pixel
    deg_per_pixel = scale / 111320.0  # xấp xỉ tại xích đạo
    nx = int((maxx - minx) / deg_per_pixel)
    ny = int((maxy - miny) / deg_per_pixel)
    
    total_bytes = nx * ny * n_bands * 8  # float64
    if total_bytes <= max_bytes:
        return 1, 1, minx, miny, maxx, maxy
    
    n_tiles = math.ceil(total_bytes / max_bytes)
    cols = math.ceil(math.sqrt(n_tiles * (maxx - minx) / max((maxy - miny), 0.001)))
    rows = math.ceil(n_tiles / max(cols, 1))
    cols = max(cols, 1)
    rows = max(rows, 1)
    return rows, cols, minx, miny, maxx, maxy

def export_image_local(image, roi, filename, out_dir, scale=30, crs="EPSG:4326"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    # Bắt buộc xóa cache cũ nếu có để tránh geemap bỏ qua không tải dữ liệu mới
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass
            
    # 🌟 KHẮC PHỤC: Khóa cứng Lưới Pixel (Grid) vào gốc tọa độ của Hệ quy chiếu (CRS).
    # Việc này chặn việc GEE tự tạo Grid neo theo Góc của từng Tile, giúp các Pixel
    # của tất cả các lớp (NDVI, DEM...) và các mảnh Tile nối nhau chính xác 100%, không bị lệch pixel.
    image = image.reproject(crs=crs, scale=scale)
    
    # Thử tải trực tiếp trước
    try:
        geemap.ee_export_image(
            image, 
            filename=out_path, 
            scale=scale, 
            region=roi, 
            crs=crs,
            file_per_band=False
        )
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception as e:
        err_msg = str(e)
        if "must be less than or equal to" not in err_msg and "Total request size" not in err_msg:
            st.error(f"Lỗi khi tải {filename}: {err_msg}")
            return None
    
    # === CHẾ ĐỘ TILED DOWNLOAD (Vùng lớn) ===
    st.info(f"⬇️ Vùng quá lớn cho tải 1 lần. Đang chia ô lưới tự động cho **{filename}**...")
    
    try:
        # Đếm bands
        n_bands = image.bandNames().size().getInfo()
        rows, cols, minx, miny, maxx, maxy = _estimate_tile_grid(roi, scale, n_bands)
        
        st.write(f"📐 Chia thành lưới **{rows}×{cols}** = {rows*cols} tile(s)")
        
        dx = (maxx - minx) / cols
        dy = (maxy - miny) / rows
        
        tile_dir = os.path.join(out_dir, f"_tiles_{os.path.splitext(filename)[0]}")
        os.makedirs(tile_dir, exist_ok=True)
        
        tile_paths = []
        progress = st.progress(0, text="Đang tải tiles...")
        total_tiles = rows * cols
        
        for r in range(rows):
            for c in range(cols):
                tile_idx = r * cols + c
                tile_minx = minx + c * dx
                tile_maxx = minx + (c + 1) * dx
                tile_miny = miny + r * dy
                tile_maxy = miny + (r + 1) * dy
                
                tile_roi = ee.Geometry.Rectangle([tile_minx, tile_miny, tile_maxx, tile_maxy])
                tile_image = image.clip(tile_roi)
                tile_name = f"tile_{r}_{c}.tif"
                tile_path = os.path.join(tile_dir, tile_name)
                
                try:
                    geemap.ee_export_image(
                        tile_image,
                        filename=tile_path,
                        scale=scale,
                        region=tile_roi,
                        crs=crs,
                        file_per_band=False
                    )
                    if os.path.exists(tile_path) and os.path.getsize(tile_path) > 0:
                        tile_paths.append(tile_path)
                except Exception as te:
                    st.warning(f"⚠️ Tile [{r},{c}] thất bại: {te}")
                
                progress.progress((tile_idx + 1) / total_tiles, text=f"Tile {tile_idx+1}/{total_tiles}")
        
        progress.empty()
        
        if not tile_paths:
            st.error("❌ Không tải được tile nào!")
            return None
        
        # === GỘP TILES BẰNG RASTERIO ===
        import rasterio
        from rasterio.merge import merge
        
        datasets = [rasterio.open(tp) for tp in tile_paths]
        merged, merged_transform = merge(datasets)
        
        # Lấy metadata từ tile đầu tiên
        meta = datasets[0].meta.copy()
        meta.update(
            height=merged.shape[1],
            width=merged.shape[2],
            transform=merged_transform
        )
        
        for ds in datasets:
            ds.close()
        
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(merged)
        
        # Dọn dẹp tiles
        import shutil
        try:
            shutil.rmtree(tile_dir)
        except:
            pass
        
        st.success(f"✅ Gộp {len(tile_paths)} tiles → **{filename}** thành công!")
        return out_path
        
    except Exception as e:
        st.error(f"Lỗi Tiled Download {filename}: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def fetch_dem_slope(roi, out_dir, scale=30, crs="EPSG:4326"):
    with st.spinner("Đang xử lý DEM và bản đồ độ dốc (Slope)..."):
        dataset = ee.Image('USGS/SRTMGL1_003')
        dem = dataset.select('elevation').clip(roi)
        slope = ee.Terrain.slope(dem).clip(roi)
        
        dem_path = export_image_local(dem, roi, 'DEM.tif', out_dir, scale, crs)
        slope_path = export_image_local(slope, roi, 'Slope.tif', out_dir, scale, crs)
        return dem_path, slope_path

def fetch_twi(roi, out_dir, scale=30, crs="EPSG:4326"):
    """Tính Topographic Wetness Index (TWI) từ DEM.
    TWI = ln(a / tan(b)), trong đó:
      a = Diện tích tích lũy dòng chảy (Flow Accumulation)
      b = Độ dốc (Slope) tính bằng radian
    TWI là biến địa hình được trích dẫn nhiều nhất trong nghiên cứu RK cho đất nông nghiệp
    vì nó phản ánh dòng chảy tích lũy nước — yếu tố quyết định phân bố N, P, K.
    """
    with st.spinner("Đang tính toán Chỉ số Ẩm ướt Địa hình (TWI)..."):
        dem = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(roi)
        
        # Tính slope theo radian
        slope_rad = ee.Terrain.slope(dem).multiply(3.14159265).divide(180)
        # Giới hạn slope tối thiểu 0.001 để tránh chia cho 0 ở vùng bằng phẳng
        slope_safe = slope_rad.max(0.001)
        
        # Tính Flow Accumulation xấp xỉ bằng Contributing Area
        # Sử dụng MERIT Hydro flow accumulation dataset (độ phân giải ~90m)
        flow_acc = (ee.Image('MERIT/Hydro/v1_0_1')
                    .select('upg')  # Upstream pixels
                    .clip(roi))
        
        # Quy đổi số pixel thượng nguồn thành diện tích (m²)
        pixel_area = ee.Image.pixelArea()
        contributing_area = flow_acc.multiply(pixel_area).max(1)
        
        # TWI = ln(a / tan(b))
        twi = contributing_area.divide(slope_safe.tan()).log().rename('TWI')
        
        twi_path = export_image_local(twi, roi, 'TWI.tif', out_dir, scale, crs)
        return twi_path

def mask_s2_csplus(image):
    """
    Sử dụng Google Cloud Score+ (CS+) để khử mây và bóng râm siêu việt.
    Ngưỡng cs_cdf >= 0.60 (hạ chuẩn để cứu vãn lỗi Null ở vùng khí hậu nhiệt đới nhiều mây).
    """
    cs_cdf = image.select('cs_cdf')
    mask = cs_cdf.gte(0.60)
    return image.updateMask(mask).divide(10000)

def fetch_ndvi(roi, start_date, end_date, out_dir, scale=10, crs="EPSG:4326"):
    with st.spinner(f"Đang phân tích NDVI Siêu Trí tuệ (Cloud Score+) từ {start_date} đến {end_date}..."):
        # 1. Gọi dữ liệu Ảnh vệ tinh Hệ đa sắc
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(roi)
                   .filterDate(start_date, end_date)
                   # Nới lỏng đầu vào để rà qua cả những tấm ảnh dính mây mờ
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)))
                   
        # 2. Gọi bộ dữ liệu Phân tích Đám mây (AI Model của GEE)
        cs_plus_collection = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
                   .filterBounds(roi)
                   .filterDate(start_date, end_date))
                   
        # 3. Khóa 2 bộ dữ liệu bằng linkCollection và xén bỏ mây
        dataset = (s2_collection.linkCollection(cs_plus_collection, ['cs_cdf'])
                   .map(mask_s2_csplus))
        
        # 4. Ép thời gian: Chọn Pixel trung vị sáng nhất, hoàn hảo hóa lỗ hổng do mây
        median_img = dataset.median().clip(roi)
        
        # 5. Phóng công thức NDVI
        ndvi = median_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        ndvi_path = export_image_local(ndvi, roi, 'NDVI.tif', out_dir, scale, crs)
        return ndvi_path

def fetch_chirps(roi, start_date, end_date, out_dir, scale=30, crs="EPSG:4326"):
    with st.spinner("Đang tải dữ liệu lượng mưa CHIRPS..."):
        dataset = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                   .filterBounds(roi)
                   .filterDate(start_date, end_date))
        
        # Đặt lại Projection gốc (5566m) do hàm sum() làm mất lưới, sau đó nội suy bilinear trơn tru xuống 30m
        total_precip = (dataset.sum()
                        .rename('Precipitation')
                        .setDefaultProjection(crs='EPSG:4326', scale=5566)
                        .resample('bilinear')
                        .clip(roi))
        
        precip_path = export_image_local(total_precip, roi, 'CHIRPS.tif', out_dir, scale, crs)
        return precip_path

def fetch_multitemporal_ndvi_stack(roi_fc, roi_box, target_year, out_dir, scale=10):
    """
    Tải Multi-temporal NDVI Stack (12 bands) từ Sentinel-2 + Cloud Score+.
    Dùng **chính xác cùng phương pháp** lọc mây AI (cs_cdf) như Bước 1,
    tránh nhiễu mây/bóng mây mà Sentinel-2 raw gặp phải.
    4 quý × 3 năm = 12 bands → Bắt nhịp sinh trưởng mía theo mùa vụ.
    """
    with st.spinner(f"Đang hút 12 Băng tần NDVI đa thời gian (Pivot {target_year}) với Cloud Score+..."):
        try:
            quarters = [
                ('Q1', '01-01', '03-31'),
                ('Q2', '04-01', '06-30'),
                ('Q3', '07-01', '09-30'),
                ('Q4', '10-01', '12-31'),
            ]
            
            years = [target_year, target_year - 1, target_year - 2]
            bands_list = []
            
            for year in years:
                for q_name, start_md, end_md in quarters:
                    start_date = f"{year}-{start_md}"
                    end_date = f"{year}-{end_md}"
                    
                    # Bước 1: Sentinel-2 SR (nới lỏng cloud 80% giống fetch_ndvi)
                    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(roi_box)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)))
                    
                    # Bước 2: Cloud Score+ AI Model (chính xác như fetch_ndvi)
                    cs_plus = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
                        .filterBounds(roi_box)
                        .filterDate(start_date, end_date))
                    
                    # Bước 3: Link + Mask mây bằng cs_cdf (ngưỡng 0.60)
                    clean_s2 = s2.linkCollection(cs_plus, ['cs_cdf']).map(mask_s2_csplus)
                    
                    # Bước 4: Median composite (lấp lỗ hổng mây)
                    median_img = clean_s2.median()
                    
                    ndvi = median_img.normalizedDifference(['B8', 'B4']).rename(f'Y{year}_{q_name}')
                    bands_list.append(ndvi)
            
            # Gộp 12 bands thành 1 image
            multi_ndvi = bands_list[0]
            for b in bands_list[1:]:
                multi_ndvi = multi_ndvi.addBands(b)
            
            multi_ndvi = multi_ndvi.clip(roi_fc)
            
            file_name = f"NDVI_MultiTemporal_Y{target_year}.tif"
            out_path = export_image_local(multi_ndvi, roi_box, file_name, out_dir, scale=scale, crs="EPSG:4326")
            return out_path
            
        except Exception as e:
            st.error(f"Lỗi truy xuất NDVI đa thời gian: {e}")
            return None
