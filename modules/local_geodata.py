import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import os
import json
import numpy as np

def mask_raster_with_vector(raster_path, geojson_path, buffer_distance=0.1):
    """
    Cắt gọt viền của một GeoTIFF bám chính xác vào đa giác GeoJSON (Irregular Polygon Masking).
    Có thể mở rộng bằng buffer_distance (đơn vị của CRS) để giữ lại các pixel vùng ven.
    """
    import geopandas as gpd
    
    with rasterio.open(raster_path) as src:
        crs = src.crs
        out_meta = src.meta.copy()
        
    gdf = gpd.read_file(geojson_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
        
    # Thêm buffer để triệt tiêu sai số GDAL hoặc cố tình giữ lại vùng ven
    if buffer_distance > 0:
        # join_style=2 (mitre) hoặc 1 (round). Dùng round để viền mịn hơn
        buffered_series = gdf.geometry.buffer(buffer_distance, join_style=1)
        geoms = buffered_series.values
    else:
        geoms = gdf.geometry.buffer(0.01).values
    
    with rasterio.open(raster_path) as src:
        # Force strict nodata representation dynamically by type
        nd_val = out_meta.get('nodata')
        if nd_val is None or nd_val == 0:
            dtype_str = str(out_meta.get('dtype', 'float32')).lower()
            if 'uint8' in dtype_str:
                nd_val = 255
            elif 'uint16' in dtype_str:
                nd_val = 65535
            else:
                nd_val = -9999.0
            
        out_image, out_transform = mask(src, geoms, crop=False, filled=True, nodata=nd_val, all_touched=True)
        
    out_meta.update({
        "transform": out_transform,
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "nodata": nd_val
    })
    
    with rasterio.open(raster_path, "w", **out_meta) as dest:
        dest.write(out_image)


def sync_raster_to_master(source_path, master_path, out_path, is_categorical=True):
    """
    Ép tệp ngoại lai về Không gian Master Raster, và kế thừa bộ lọc Xén Viền (Mask) của Master.
    """
    with rasterio.open(master_path) as master:
        dest_transform = master.transform
        dest_crs = master.crs
        dest_width = master.width
        dest_height = master.height
        master_data = master.read(1)
        m_nodata = master.nodata if master.nodata is not None else 0
        
    resampling_method = Resampling.nearest if is_categorical else Resampling.bilinear
    
    with rasterio.open(source_path) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dest_crs,
            'transform': dest_transform,
            'width': dest_width,
            'height': dest_height,
            'nodata': 0
        })
        
        # Mảng Resample
        destination_array = np.zeros((dest_height, dest_width), dtype=kwargs['dtype'])
        
        reproject(
            source=rasterio.band(src, 1),
            destination=destination_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dest_transform,
            dst_crs=dest_crs,
            resampling=resampling_method
        )
        
        # Áp chặn Xén Viền (Mask Inherit)
        destination_array[master_data == m_nodata] = 0
        
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            dst.write(destination_array, 1)
    return out_path


def rasterize_vector_to_master(geojson_path, attribute_col, master_raster_path, out_raster_path):
    """
    Nướng Vector vào Master Raster, và tự cắt viền rỗng theo Master Raster (Inherit Mask).
    """
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize
    
    gdf = gpd.read_file(geojson_path)
    
    with rasterio.open(master_raster_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs
        master_data = src.read(1)
        m_nodata = src.nodata if src.nodata is not None else 0
        
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
        
    if gdf[attribute_col].dtype == 'object' or pd.api.types.is_string_dtype(gdf[attribute_col]):
        labels, uniques = pd.factorize(gdf[attribute_col])
        gdf['raster_val'] = labels + 1 
        legend_dict = {str(val): int(code) + 1 for code, val in enumerate(uniques)}
        legend_path = out_raster_path.replace('.tif', '_legend.json')
        with open(legend_path, 'w', encoding='utf-8') as f:
            json.dump(legend_dict, f, ensure_ascii=False, indent=2)
        val_col = 'raster_val'
    else:
        val_col = attribute_col
        gdf[val_col] = gdf[val_col].fillna(0)
        
    shapes = ((geom.buffer(0.1), val) for geom, val in zip(gdf.geometry, gdf[val_col]) if geom is not None)
    
    burned = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        fill=0,
        transform=transform,
        all_touched=True,
        dtype=rasterio.int16
    )
    
    # Áp chặn Xén Viền từ DEM/Master
    burned[master_data == m_nodata] = 0
    
    meta.update(
        dtype=rasterio.int16,
        count=1,
        nodata=0,
        compress='lzw'
    )
    
    with rasterio.open(out_raster_path, 'w', **meta) as out:
        out.write(burned, 1)

def align_raster_to_sacred_grid(raster_path, dst_crs, dst_transform, dst_width, dst_height, is_categorical=False):
    """
    Kéo giãn (Warp) và ép cứng mọi raster lưới mờ từ GEE vào Lưới Thiêng (Sacred Grid) 
    để đảm bảo 100% pixel ngang bằng sổ thẳng không lệch 1 milimet.
    """
    from rasterio.warp import reproject, Resampling
    import numpy as np
    
    with rasterio.open(raster_path) as src:
        kwargs = src.meta.copy()
        
        # Override nodata using safe values per dtype
        dtype_str = str(kwargs['dtype']).lower()
        if 'uint8' in dtype_str:
            nd_val = 255
        elif 'uint16' in dtype_str:
            nd_val = 65535
        else:
            nd_val = -9999.0
        kwargs.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': nd_val
        })
        
        # Fill array with default nodata value
        destination_array = np.full((dst_height, dst_width), nd_val, dtype=kwargs['dtype'])
        resampling = Resampling.nearest if is_categorical else Resampling.bilinear
        
        reproject(
            source=rasterio.band(src, 1),
            destination=destination_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling
        )
        
    with rasterio.open(raster_path, 'w', **kwargs) as dst:
        dst.write(destination_array, 1)
