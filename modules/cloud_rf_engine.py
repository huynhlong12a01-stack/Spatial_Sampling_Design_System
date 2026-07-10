import ee
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import requests
import streamlit as st
import time

def grid_bounds(minx, miny, maxx, maxy, step_km=15, resolution=10):
    """Chia Bounding Box lớn thành các ô chữ nhật nhỏ."""
    res_deg = resolution / 111320.0
    
    # Nhích step_deg để nó vừa chẵn với độ phân giải (chống lệch Grid)
    step_deg_raw = step_km / 111.0
    step_deg = np.round(step_deg_raw / res_deg) * res_deg
    if step_deg == 0:
        step_deg = res_deg * 10
        
    # Snap min/max để lưới các ô vuông hoàn toàn thẳng hàng theo pixel
    minx_snap = np.floor(minx / res_deg) * res_deg
    miny_snap = np.floor(miny / res_deg) * res_deg
    maxx_snap = np.ceil(maxx / res_deg) * res_deg
    maxy_snap = np.ceil(maxy / res_deg) * res_deg
    
    x_coords = np.arange(minx_snap, maxx_snap + step_deg/2, step_deg)
    y_coords = np.arange(miny_snap, maxy_snap + step_deg/2, step_deg)
    
    tiles = []
    for i in range(len(x_coords)-1):
        for j in range(len(y_coords)-1):
            if (x_coords[i+1] - x_coords[i] > res_deg/2) and (y_coords[j+1] - y_coords[j] > res_deg/2):
                tiles.append([x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1]])
    return tiles

def run_gee_pipeline(gdf_roi, gdf_gt, target_year, out_boundary_path, resolution=10, status=None):
    import geemap
    
    # 0. CHUẨN BỊ INPUT
    # ==========================
    st_date_24 = f"{target_year}-01-01"
    ed_date_24 = f"{target_year}-12-31"
    
    st_date_2yr = f"{target_year-1}-01-01"
    ed_date_2yr = f"{target_year}-12-31"
    
    # Ép dạng Địa lý gốc (Polygon) thay vì xài Bounding Box tốn diện tích
    roi_fc = geemap.geopandas_to_ee(gdf_roi.to_crs("EPSG:4326"))
    roi_geom = roi_fc.geometry()
    
    # Tính Bounding Box dùng cho việc chia Lưới (Tiling) ở phần sau
    overall_bounds = gdf_roi.to_crs("EPSG:4326").total_bounds
    minx, miny, maxx, maxy = overall_bounds
    
    # Ép GT thành List[ee.Feature]
    features_list = []
    # Khử rác rỗng (NaN) từ CSV nếu có
    valid_gt = gdf_gt[gdf_gt.is_valid & ~gdf_gt.is_empty].dropna(subset=['geometry'])
    for _, row in valid_gt.to_crs("EPSG:4326").iterrows():
        if pd.notna(row.geometry.x) and pd.notna(row.geometry.y):
            coords = [row.geometry.x, row.geometry.y]
            features_list.append(ee.Feature(ee.Geometry.Point(coords), {'class': 1}))
            
    sugarcane_fc = ee.FeatureCollection(features_list)
    if status: status.write(f"- ✔️ [Bước 1/5] Đã nén thành công Vùng quan tâm (Polygon ROI) và {len(valid_gt)} Tọa độ Mía Ground Truth.")
    
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select('Map').clip(roi_geom)
    CROPLAND = 40
    croplandMask = worldcover.eq(CROPLAND).selfMask()
    
    # ==========================
    # 1. TẠO TẬP ĐIỂM (TABULAR SAMPLING POINTS)
    # ==========================
    # Xả mù ngẫu nhiên lấy tọa độ. Ép giới hạn an toàn 5000 điểm để chứa lượng Hard Negatives.
    overall_bounds_geom = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    max_blind_points = min(len(valid_gt) * 5, 5000)
    blind_random_pts = ee.FeatureCollection.randomPoints(
        region=overall_bounds_geom,
        points=max_blind_points,
        seed=42
    ).map(lambda f: f.set('class', 0))
    
    # Gộp mẫu Dương tính (từ User) và mẫu ngẫu nhiên
    trainingPts = sugarcane_fc.merge(blind_random_pts)
        
    if status: status.write(f"- ✔️ [Bước 2/5] Đã xả {max_blind_points} điểm mù ngẫu nhiên thành công.")
    
    # ==========================
    # 2. HÀM CORE: XỬ LÝ LỌC CĂN BẢN (Gắn Cloud Score+)
    # ==========================
    # Kết hợp filter của User với CLOUD_SCORE_PLUS
    def maskS2_CSPlus(img):
        cs_cdf = img.select('cs_cdf')
        # Loại sương mù và mây
        good = cs_cdf.gte(0.60)
        # Bóp méo tính toán: Chỉ tính toán trên Cropland (Early Masking)
        return img.updateMask(good).updateMask(croplandMask).divide(10000).copyProperties(img, ['system:time_start'])

    def addS2Indices(img):
        ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI')
        evi  = img.expression(
            '2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))',
            {'NIR': img.select('B8'), 'RED': img.select('B4'), 'BLUE': img.select('B2')}
        ).rename('EVI')
        ndmi = img.normalizedDifference(['B8','B11']).rename('NDMI')
        ndre = img.normalizedDifference(['B8','B5']).rename('NDRE')
        nbr2 = img.normalizedDifference(['B11','B12']).rename('NBR2')
        ndbi = img.normalizedDifference(['B11','B8']).rename('NDBI')
        
        return img.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']) \
                  .addBands([ndvi, evi, ndmi, ndre, nbr2, ndbi]) \
                  .copyProperties(img, ['system:time_start'])

    # Query Base S2
    s2_base = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(roi_geom) \
                .filterDate(st_date_2yr, ed_date_2yr) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                
    cs_base = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED') \
                .filterBounds(roi_geom) \
                .filterDate(st_date_2yr, ed_date_2yr)
                
    s2_2yr = s2_base.linkCollection(cs_base, ['cs_cdf']).map(maskS2_CSPlus).map(addS2Indices)
    s2_current = s2_2yr.filterDate(st_date_24, ed_date_24)
    if status: status.write(f"- ⏳ [Bước 3/5] Đang xếp tầng Quang phổ 2 năm (Sentinel-2 Khử Mây, S1 SAR, Chỉ số Khí hậu Thống kê)...")

    # ==========================
    # 3. TINH CHẾ CÁC BIẾN (PERCENTILES)
    # ==========================
    idxBands = ['NDVI','EVI','NDMI','NDRE','NBR2','NDBI']
    
    # Safe Helper trong GEE cho rỗng Image
    def emptyImg(bandList):
        return ee.Image.constant(ee.List.repeat(0, len(bandList))).rename(bandList).updateMask(ee.Image(0))
        
    statsReducer = ee.Reducer.percentile([10,50,90])
    outNamesNoPrefix = []
    for b in idxBands:
        outNamesNoPrefix.extend([f"{b}_p10", f"{b}_p50", f"{b}_p90"])
        
    def outNames_2yr(n): return f"{n}_3yr"
    def outNames_tgt(n): return f"{n}_{target_year}"

    # Lấy features 3 yr
    s2Stats2yr = ee.Algorithms.If(
        s2_2yr.size().gt(0),
        s2_2yr.select(idxBands).reduce(statsReducer).rename([outNames_2yr(n) for n in outNamesNoPrefix]),
        emptyImg([outNames_2yr(n) for n in outNamesNoPrefix])
    )
    s2Stats2yr = ee.Image(s2Stats2yr)

    # Lấy features current yr
    s2StatsTgt = ee.Algorithms.If(
        s2_current.size().gt(0),
        s2_current.select(idxBands).reduce(statsReducer).rename([outNames_tgt(n) for n in outNamesNoPrefix]),
        emptyImg([outNames_tgt(n) for n in outNamesNoPrefix])
    )
    s2StatsTgt = ee.Image(s2StatsTgt)
    
    # NDVI Amp
    ndviP90 = s2_current.select(['NDVI']).reduce(ee.Reducer.percentile([90])).rename(f'NDVIP90')
    ndviP10 = s2_current.select(['NDVI']).reduce(ee.Reducer.percentile([10])).rename(f'NDVIP10')
    ndviAmp = ndviP90.subtract(ndviP10).rename('NDVI_amp')
    
    # 3.1. FEATURE ENGINEERING: GLCM Texture
    glcm_img = ndviP90.add(1).multiply(100).toInt32().glcmTexture(size=3)
    glcm_var = glcm_img.select('NDVIP90_var').rename('GLCM_Var')
    glcm_ent = glcm_img.select('NDVIP90_ent').rename('GLCM_Ent')
    
    # 3.2. FEATURE ENGINEERING: Harmonic Analysis (Fourier Pheonology)
    def addTime(img):
        date = img.date()
        years = date.difference(ee.Date('1970-01-01'), 'year')
        timeRadians = ee.Image(years.multiply(2 * np.pi))
        return img.addBands(ee.Image.constant(1).rename('constant')) \
                  .addBands(timeRadians.cos().rename('cos')) \
                  .addBands(timeRadians.sin().rename('sin'))
                  
    s2_harmonic = s2_current.map(addTime)
    independents = ee.List(['constant', 'cos', 'sin'])
    trend = s2_harmonic.select(independents.add('NDVI')) \
                       .reduce(ee.Reducer.linearRegression(independents.length(), 1))
    
    coefs = trend.select('coefficients').arrayProject([0]).arrayFlatten([independents])
    cos_c = coefs.select('cos')
    sin_c = coefs.select('sin')
    
    harmonicPhase = sin_c.atan2(cos_c).rename('Harmonic_Phase')
    harmonicAmp = cos_c.hypot(sin_c).rename('Harmonic_Amp')

    # Seasonal Target Year
    dryIC = s2_current.filterDate(f"{target_year}-01-01", f"{target_year}-04-30")
    wetIC = s2_current.filterDate(f"{target_year}-07-01", f"{target_year}-10-31")
    
    seasonNeed = ['NDVI','NDMI','NDRE','NDBI']
    s2Dry = ee.Image(ee.Algorithms.If(dryIC.size().gt(0), dryIC.select(seasonNeed).median(), emptyImg(seasonNeed)))
    s2Wet = ee.Image(ee.Algorithms.If(wetIC.size().gt(0), wetIC.select(seasonNeed).median(), emptyImg(seasonNeed)))
    
    seasonFeat = ee.Image.cat([
        s2Dry.rename([f"{x}_dry" for x in seasonNeed]),
        s2Wet.rename([f"{x}_wet" for x in seasonNeed])
    ])
    
    s2ValidCount = s2_current.select('NDVI').count().rename('S2_validCount')

    # ==========================
    # 4. SENTINEL-1 SAR (Giới hạn nhẹ)
    # ==========================
    def prepS1(img):
        img_masked = img.updateMask(croplandMask)
        vv = img_masked.select('VV')
        vh = img_masked.select('VH')
        diff = vv.subtract(vh).rename('VVminusVH')
        return ee.Image.cat([vv.rename('VV'), vh.rename('VH'), diff])

    s1_base = ee.ImageCollection('COPERNICUS/S1_GRD') \
                .filterBounds(roi_geom) \
                .filter(ee.Filter.eq('instrumentMode','IW')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH')) \
                .select(['VV','VH']) \
                .map(lambda img: img.log10().multiply(10.0)) \
                .map(prepS1)
                
    s1_2yr = s1_base.filterDate(st_date_2yr, ed_date_2yr)
    s1_tgt = s1_base.filterDate(st_date_24, ed_date_24)
    
    s1Reducer = ee.Reducer.percentile([25,50,75])
    s1OutNames_base = ['VV_p25','VV_p50','VV_p75', 'VH_p25','VH_p50','VH_p75', 'VVminusVH_p25','VVminusVH_p50','VVminusVH_p75']
    
    # Late Smoothing function & RVI Derivation
    def apply_s1_smoothing(img):
        img_smooth = ee.Image(img).focal_median(30, 'circle', 'meters').updateMask(croplandMask)
        
        vv_lin = ee.Image(10.0).pow(img_smooth.select('.*VV.*').divide(10.0))
        vh_lin = ee.Image(10.0).pow(img_smooth.select('.*VH.*').divide(10.0))
        rvi_bands = vh_lin.multiply(4).divide(vv_lin.add(vh_lin))
        
        def renameRVI(bn): return ee.String(bn).replace('VH_','RVI_')
        rvi_bands = rvi_bands.rename(rvi_bands.bandNames().map(renameRVI))
        
        return img_smooth.addBands(rvi_bands)

    s1Stats2yr = ee.Image(ee.Algorithms.If(
        s1_2yr.size().gt(0),
        apply_s1_smoothing(s1_2yr.reduce(s1Reducer).rename([f"S1_2yr_{n}" for n in s1OutNames_base])),
        emptyImg([f"S1_2yr_{n}" for n in s1OutNames_base])
    ))

    s1StatsTgt = ee.Image(ee.Algorithms.If(
        s1_tgt.size().gt(0),
        apply_s1_smoothing(s1_tgt.reduce(s1Reducer).rename([f"S1_{target_year}_{n}" for n in s1OutNames_base])),
        emptyImg([f"S1_{target_year}_{n}" for n in s1OutNames_base])
    ))
    
    # ==========================
    # 5. TOPO & TWI
    # ==========================
    dem = ee.Image('USGS/SRTMGL1_003').rename('elev')
    slope = ee.Terrain.slope(dem).rename('slope')
    
    slope_rad = slope.multiply(np.pi / 180.0)
    tan_slope = slope_rad.tan().max(0.001)
    upa = ee.Image("MERIT/Hydro/v1_0_1").select('upa')
    twi = upa.divide(tan_slope).log().rename('TWI')
    
    # ==========================
    # 6. FEATURE STACK GỘP VÀ HUẤN LUYỆN ML
    # ==========================
    featureImage = ee.Image.cat([
        s2Stats2yr, s2StatsTgt,
        ndviP90, ndviP10, ndviAmp,
        seasonFeat,
        harmonicPhase, harmonicAmp,
        glcm_var, glcm_ent,
        s1Stats2yr, s1StatsTgt,
        dem, slope, twi,
        s2ValidCount,
        worldcover.eq(40).rename('is_crop') # Bắn label này vào để lọc ngầm
    ]).unmask(0)
    
    bands_with_crop = featureImage.bandNames()
    bands = bands_with_crop.remove('is_crop') # Chặn is_crop vào Model Data
    
    # Kéo Vector cắm ống hút trích xuất dữ liệu mảng
    training = featureImage.sampleRegions(
        collection=trainingPts,
        properties=['class'],
        scale=resolution,
        tileScale=16
    ).filter(ee.Filter.gt('S2_validCount', 0))
    
    # KHAI THÁC MẪU ÂM TÍNH 1:3 & HARD NEGATIVES
    valid_positives = training.filter(ee.Filter.eq('class', 1))
    valid_negatives = training.filter(ee.Filter.eq('class', 0)).filter(ee.Filter.eq('is_crop', 1))
    
    hard_negatives = valid_negatives.filter(ee.Filter.gte('NDVIP90', 0.60))
    easy_negatives = valid_negatives.filter(ee.Filter.lt('NDVIP90', 0.60))
    
    nPos = valid_positives.size()
    nHalfNeg = nPos.multiply(1.5).round() # Total negative = Pos * 3. So Each half = Pos * 1.5
    
    sel_hard = hard_negatives.randomColumn('r_h').sort('r_h').limit(nHalfNeg)
    sel_easy = easy_negatives.randomColumn('r_e').sort('r_e').limit(nHalfNeg)
    all_negatives = sel_hard.merge(sel_easy)
    
    fullDataSet = valid_positives.merge(all_negatives).randomColumn('split_rand')
    
    # 80/20 Train-Val Split
    trainSet = fullDataSet.filter(ee.Filter.lt('split_rand', 0.80))
    valSet = fullDataSet.filter(ee.Filter.gte('split_rand', 0.80))
    
    if status: status.write(f"- ⏳ [Bước 4/5] GEE đang nén và rút trích Lịch sử Quang phổ tại {len(valid_gt) + max_blind_points} tọa độ neo. (Mất 1 Cửa Sổ Timeout ~ 3 Phút)...")
    
    # FORCE EXECUTION CHUNK 1: Materialize Data
    try:
        total_training_samples = trainSet.size().getInfo()
        if status: status.write(f"- ✔️ [Bước 4/5] Trích xuất xong (Tránh Timeout v1)! GEE đã hợp nhất và cân bằng thành {total_training_samples} mẫu Hợp Lệ Cốt Lõi.")
    except Exception as e:
        if status: status.write(f"- ❌ Lỗi Timeout Tầng Sâu (Khi cắm vòi trích xuất): {e}")
        return None
    
    if status: status.write(f"- ⏳ [Bước Tăng cường] Đang ra lệnh GEE nhồi Ma Trận vào 150 Cây Quyết Định (Random Forest). (Mất 1 Cửa Sổ Timeout)...")
    
    # Build Classifier
    vps = ee.Number(bands.size()).sqrt().round().max(2)
    rf = ee.Classifier.smileRandomForest(
        numberOfTrees=150,
        variablesPerSplit=vps,
        minLeafPopulation=5,
        bagFraction=0.7,
        seed=7
    ).train(
        features=trainSet,
        classProperty='class',
        inputProperties=bands
    )
    
    # FORCE EXECUTION CHUNK 2: Materialize Model & Dynamic Thresholding
    try:
        val_predictions = valSet.classify(rf.setOutputMode('PROBABILITY'), "cane_prob").getInfo()
        if status: status.write(f"- ✔️ [Bước Tăng cường] Nền tảng Tri thức AI đã Cached cấu trúc. Đang dò ngưỡng Siêu Tham Số (Dynamic Thresholding)...")
        
        y_true = []
        y_probs = []
        for feat in val_predictions['features']:
            props = feat['properties']
            y_true.append(props['class'])
            y_probs.append(props.get('cane_prob', 0))
            
        best_f1 = 0
        best_th = 0.55
        for t_int in range(40, 75, 2):
            th = t_int / 100.0
            tp = sum(1 for yt, yp in zip(y_true, y_probs) if yp >= th and yt == 1)
            fp = sum(1 for yt, yp in zip(y_true, y_probs) if yp >= th and yt == 0)
            fn = sum(1 for yt, yp in zip(y_true, y_probs) if yp < th and yt == 1)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                
        if status: status.write(f"- ✔️ [Bước Siêu Tham Cương] Độ chính xác F1-Score Tập Val đạt {round(best_f1*100, 2)}% tại Ngưỡng cắt động >= {best_th}.")
        dynamic_threshold = best_th
    except Exception as e:
        if status: status.write(f"- ⚠️ Lỗi dò ngưỡng Siêu Tham Số, hệ thống lùi về ngưỡng gốc 0.55. Lỗi: {e}")
        dynamic_threshold = 0.55
    
    # ==========================
    # 7. INFERENCE VÀ XỬ LÝ ẢNH NHỊ PHÂN & OBIA SUPERPIXELS
    # ==========================
    featureForPredict = featureImage.updateMask(worldcover.eq(CROPLAND))
    
    # 7.1 Pixel-based Probability
    probRaw = featureForPredict.classify(rf.setOutputMode('PROBABILITY')).rename('cane_prob')
    
    # 7.2 Post-Classification OBIA (SNIC)
    if status: status.write(f"- ⏳ [Bước Siêu Tăng Cường] Đang chia lưới Siêu Điểm Ảnh (SNIC Superpixels) để khử hoàn toàn Nhiễu Muối Tiêu...")
    
    # Define Anchor Composite: NIR(B8), RED(B4), GREEN(B3) + NDVI_P90
    # Truyền luôn probRaw vào Anchor. SNIC sẽ tự động nội suy Trung bình Cộng (mean) của toàn bộ các Band đầu vào!
    # Không cần xài reduceConnectedComponents nặng nề nữa!
    snic_anchor = s2_current.select(['B8', 'B4', 'B3']).median().rename(['NIR', 'RED', 'GREEN']).addBands(ndviP90).addBands(probRaw)
    snic_anchor_masked = snic_anchor.updateMask(worldcover.eq(CROPLAND))
    
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=snic_anchor_masked,
        size=10,
        compactness=0,
        connectivity=8,
        neighborhoodSize=128
    )
    
    # 7.3 Extract Native OBIA Mean (Sát thủ chống OOM)
    # Lấy thẳng cane_prob_mean được SNIC tự động đúc ra từ probRaw gốc
    probOBIA = snic.select('cane_prob_mean').rename('cane_prob')
    
    # 7.4 Final Binary Object-Based Masking
    caneFinal = probOBIA.gte(dynamic_threshold).unmask(0).multiply(255).toByte().rename('cane')
    
    # ==========================
    # 8. MACRO-TILING & LOCAL VECTORIZATION CHỐNG SẬP API 32MB
    # ==========================
    dynamic_step_km = 15 if resolution >= 30 else 5
    
    if status: status.write(f"- ⏳ [Bước 5/5] Máy Xử lý Raster GEE đã nung Cây Xong. Tiến hành tách Lưới {dynamic_step_km}km để Tải Raster TIF Cục Bộ dưới RAM Máy tính...")
    
    tiles = grid_bounds(minx, miny, maxx, maxy, step_km=dynamic_step_km, resolution=resolution) 
    all_tifs = []
    
    progress_text = "Đang Tải Ảnh TIF Cục Bộ..."
    my_bar = st.progress(0, text=progress_text)
    
    project_dir = os.path.dirname(out_boundary_path)
    tmp_folder = os.path.join(project_dir, 'ai_rasters_output')
    os.makedirs(tmp_folder, exist_ok=True)
    
    import requests, zipfile, rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape as shapely_shape
    
    for i, t in enumerate(tiles):
        tile_geom = ee.Geometry.Rectangle(t)
        
        # Chỉ tải Raster nguyên khối hình chữ nhật vuông vức (KHÔNG xén phức tạp trên GEE)
        layer_tile = caneFinal.clip(tile_geom)
        
        success = False
        for attempt in range(3):
            try:
                url = layer_tile.getDownloadURL({
                    'scale': resolution,
                    'crs': 'EPSG:4326',
                    'region': tile_geom,
                    'format': 'GEO_TIFF'
                })
                
                r = requests.get(url, timeout=300)
                if r.status_code == 200:
                    zip_path = os.path.join(tmp_folder, f"sugarcane_grid_{i}.zip")
                    with open(zip_path, 'wb') as f:
                        f.write(r.content)
                    
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as z:
                            tif_names = [n for n in z.namelist() if n.endswith('.tif')]
                            if tif_names:
                                out_tif_name = f"sugarcane_prob_grid_{i}.tif"
                                z.extract(tif_names[0], tmp_folder)
                                extracted_path = os.path.join(tmp_folder, tif_names[0])
                                tif_path = os.path.join(tmp_folder, out_tif_name)
                                if os.path.exists(tif_path): os.remove(tif_path)
                                os.rename(extracted_path, tif_path)
                                
                                all_tifs.append(tif_path)
                                success = True
                    except zipfile.BadZipFile:
                        raw_tif_path = os.path.join(tmp_folder, f"sugarcane_prob_grid_{i}.tif")
                        with open(raw_tif_path, "wb") as f:
                            f.write(r.content)
                            
                        all_tifs.append(raw_tif_path)
                        success = True
                    
                    break # Success, clear to break nested loop
                elif r.status_code == 429: # Rate limit
                    time.sleep(5)
                else:
                    print(f"Lỗi API HTTP {r.status_code} ô lưới {i}: {r.text}")
                    time.sleep(2)
            except Exception as err:
                print(f"Lỗi tải ô lưới {i} (Lần thử {attempt+1}): {err}")
                time.sleep(2)
                
        if not success and status:
            status.write(f"⚠️ Ô lưới số {i} không thể tải sau 3 lần rớt mạng. Ranh giới có rỗng hoặc API quá tải.")
            
        my_bar.progress((i + 1) / len(tiles), text=f"{progress_text} ({i + 1}/{len(tiles)})")
        time.sleep(0.5)

    my_bar.empty()
    
    # TIẾN HÀNH GHÉP FILE MOSAIC & LỌC CHÍNH XÁC THEO POLYGON BẰNG RASTERIO (LOCAL TIF CLIP)
    if len(all_tifs) > 0:
        from rasterio.merge import merge
        from rasterio.mask import mask
        
        src_files_to_mosaic = []
        for fp in all_tifs:
            try:
                src_files_to_mosaic.append(rasterio.open(fp))
            except:
                pass
                
        if len(src_files_to_mosaic) > 0:
            mosaic, out_trans = merge(src_files_to_mosaic)
            out_meta = src_files_to_mosaic[0].meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": mosaic.shape[1],
                             "width": mosaic.shape[2],
                             "transform": out_trans})
                             
            # Nhả bộ nhớ
            for fp in src_files_to_mosaic:
                fp.close()
                
            out_tif_final = out_boundary_path.replace(".geojson", ".tif")
            
            # Xén gọt dứt điểm bằng Ranh giới Local (Clip Raster)
            gdf_roi_prj = gdf_roi.to_crs(out_meta['crs'])
            geoms = [gdf_roi_prj.geometry.unary_union]
            
            import rasterio.io
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**out_meta) as mf:
                    mf.write(mosaic)
                with memfile.open() as mf:
                    out_image, out_transform = mask(mf, geoms, crop=True)
                    out_meta.update({"height": out_image.shape[1],
                                     "width": out_image.shape[2],
                                     "transform": out_transform})
                    with rasterio.open(out_tif_final, "w", **out_meta) as dest:
                        dest.write(out_image)
                        
            return out_tif_final
            
    return None
