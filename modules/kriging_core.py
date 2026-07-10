import numpy as np
import rasterio
from rasterio.transform import from_origin
import gstools as gs
import os

def run_regression_kriging(
    raster_files,
    model, 
    vario_model_fit, 
    x_coords, y_coords, residuals,
    out_dir,
    output_filename="RK_Result.tif"
):
    """
    Thực hiện Regression Kriging toàn cục:
    1. Đọc các raster (PC hoặc covariates gốc) thành grid.
    2. Dự báo bằng Hồi quy (Trend) — model.predict() trực tiếp.
    3. Nội suy Kriging phần dư (Residuals).
    4. Cộng Trend + Kriged Residuals và lưu GeoTIFF.
    
    Lưu ý: Nếu input là PC rasters (đã PCA ở Stage 2),
    KHÔNG cần PCA transform ở đây vì rasters đã ở PC space.
    """
    meta = None
    features = []
    
    for i, path in enumerate(raster_files):
        with rasterio.open(path) as src:
            if i == 0:
                meta = src.meta.copy()
                cols, rows = np.meshgrid(np.arange(src.width), np.arange(src.height))
                T = src.transform
                xs = T.c + (cols + 0.5) * T.a + (rows + 0.5) * T.b
                ys = T.f + (cols + 0.5) * T.d + (rows + 0.5) * T.e
                
            arr = src.read(1).astype(np.float32)
            features.append(arr)
            
    # Tạo mask loại bỏ NoData
    nodata_mask = np.ones_like(features[0], dtype=bool)
    for f in features:
        nodata_mask &= (~np.isnan(f))
        if meta.get('nodata') is not None:
            nodata_mask &= (f != meta['nodata'])
            
    # Chỉ xử lý pixel hợp lệ
    X_valid = np.column_stack([f[nodata_mask] for f in features])
    xs_valid = xs[nodata_mask]
    ys_valid = ys[nodata_mask]
    
    # 1. Dự đoán Trend — model đã train trên cùng không gian (PC hoặc gốc)
    trend_pred = model.predict(X_valid)
    
    # 2. Nội suy Kriging cho phần dư
    krige = gs.krige.Ordinary(
        model=vario_model_fit,
        cond_pos=[x_coords, y_coords],
        cond_val=residuals,
        exact=True
    )
    
    krige_res, krige_var = krige(
        [xs_valid, ys_valid], return_var=True
    )
    
    # 3. Cộng gộp: Final = Trend + Kriged Residuals
    final_pred = trend_pred + krige_res
    
    # 4. Lưu kết quả ra Raster 2D
    final_raster = np.full_like(features[0], np.nan, dtype=np.float32)
    final_raster[nodata_mask] = final_pred
    
    var_raster = np.full_like(features[0], np.nan, dtype=np.float32)
    var_raster[nodata_mask] = krige_var
    
    trend_raster = np.full_like(features[0], np.nan, dtype=np.float32)
    trend_raster[nodata_mask] = trend_pred
    
    res_raster = np.full_like(features[0], np.nan, dtype=np.float32)
    res_raster[nodata_mask] = krige_res
    
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, output_filename)
    meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
    
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(final_raster, 1)
        
    var_path = os.path.join(out_dir, "Uncertainty_" + output_filename)
    with rasterio.open(var_path, 'w', **meta) as dst:
        dst.write(var_raster, 1)
        
    trend_path = os.path.join(out_dir, "Trend_" + output_filename)
    with rasterio.open(trend_path, 'w', **meta) as dst:
        dst.write(trend_raster, 1)
        
    res_path = os.path.join(out_dir, "Residual_" + output_filename)
    with rasterio.open(res_path, 'w', **meta) as dst:
        dst.write(res_raster, 1)
        
    return out_path, var_path
