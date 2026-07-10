import os
import rasterio
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

def stack_rasters(raster_paths):
    """Đọc và gộp các raster để tạo ma trận đặc trưng (Feature Matrix)"""
    features = []
    cat_features = []
    meta = None
    coords = None
    valid_mask = None
    feature_names = []
    
    for i, path in enumerate(raster_paths):
        is_cat = "Soil_Class" in os.path.basename(path) or "Categorical" in os.path.basename(path)
        
        with rasterio.open(path) as src:
            if meta is None:
                meta = src.meta.copy()
                rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing='ij')
                xs, ys = rasterio.transform.xy(src.transform, rows, cols)
                coords = np.column_stack([np.array(xs).flatten(), np.array(ys).flatten()])
            
            arr = src.read(1).flatten()
            nodata = src.nodata
            if valid_mask is None:
                if nodata is not None:
                    valid_mask = (arr != nodata) & (~np.isnan(arr))
                else:
                    valid_mask = ~np.isnan(arr)
            else:
                if nodata is not None:
                    valid_mask = valid_mask & (arr != nodata) & (~np.isnan(arr))
                else:
                    valid_mask = valid_mask & (~np.isnan(arr))
            
            if is_cat:
                cat_features.append(arr)
            else:
                features.append(arr)
                feature_names.append(os.path.basename(path).replace('.tif', ''))
    X_full = np.column_stack(features) if features else None
    X_valid = X_full[valid_mask] if X_full is not None else None
    
    X_cat_full = np.column_stack(cat_features) if cat_features else None
    X_cat_valid = X_cat_full[valid_mask] if X_cat_full is not None else None
    
    coords_valid = coords[valid_mask] if coords is not None else None
    
    return X_valid, X_cat_valid, coords_valid, meta, valid_mask, features, feature_names


def fit_pca_and_save(X_valid, valid_mask, features, meta, cov_dir, project_dir):
    """
    Fit PCA trên TOÀN BỘ raster (population-level),
    lưu PCA model + Scaler vào assets, và xuất PC rasters.
    
    Đây là nguồn PCA DUY NHẤT cho toàn dự án (Stage 2, 3, 4).
    """
    pca_dir = os.path.join(project_dir, 'pca')
    os.makedirs(pca_dir, exist_ok=True)
    
    # Bước 1: Chuẩn hoá Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)
    
    # Bước 2: PCA — giữ 95% tổng phương sai
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    n_pc = X_pca.shape[1]
    explained = pca.explained_variance_ratio_.sum() * 100
    
    # Bước 3: Lưu PCA model + Scaler (dùng lại ở Stage 3 & 4)
    pca_pack = {
        'scaler': scaler,
        'pca': pca,
        'n_components': n_pc,
        'explained_variance': explained,
        'original_features': X_valid.shape[1]
    }
    joblib.dump(pca_pack, os.path.join(pca_dir, 'pca_model.joblib'))
    
    # Bước 4: Xuất PC rasters (PC1.tif, PC2.tif...) vào thư mục pca/
    # Khôi phục ma trận 2D từ valid pixels
    first_feature = features[0]
    shape_2d = (meta['height'], meta['width']) if isinstance(first_feature, np.ndarray) and first_feature.ndim == 2 else None
    
    if shape_2d is None:
        # features đã bị flatten, khôi phục shape từ meta
        shape_2d = (meta['height'], meta['width'])
    
    pc_raster_paths = []
    for pc_idx in range(n_pc):
        pc_raster = np.full(valid_mask.shape, np.nan, dtype=np.float32)
        pc_raster[valid_mask] = X_pca[:, pc_idx].astype(np.float32)
        
        # Reshape về 2D
        pc_2d = pc_raster.reshape(shape_2d)
        
        pc_filename = f"PC{pc_idx + 1}.tif"
        pc_path = os.path.join(pca_dir, pc_filename)
        
        pc_meta = meta.copy()
        pc_meta.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
        
        with rasterio.open(pc_path, 'w', **pc_meta) as dst:
            dst.write(pc_2d, 1)
        
        pc_raster_paths.append(pc_path)
    
    pca_info = {
        'n_components': n_pc,
        'explained_variance': explained,
        'original_features': X_valid.shape[1],
        'pc_raster_paths': pc_raster_paths,
        'variance_per_pc': (pca.explained_variance_ratio_ * 100).tolist()
    }
    
    return X_pca, pca_info


def elbow_analysis(X_pca, k_min=10, k_max=100, k_step=5, progress_callback=None):
    """Phương pháp Cùi chỏ (Elbow Method) để tìm K tối ưu cho K-Means.
    
    Chạy K-Means nhiều lần với K tăng dần, đo WCSS (Within-Cluster Sum of Squares).
    Phát hiện điểm gãy (elbow) bằng thuật toán Maximum Curvature.
    
    Returns:
        k_optimal: Số K tại điểm cùi chỏ
        k_range: Danh sách K đã thử
        wcss_list: Danh sách WCSS tương ứng
        fig: Matplotlib figure của Elbow Plot
    """
    import matplotlib.pyplot as plt
    
    k_range = list(range(k_min, min(k_max + 1, len(X_pca)), k_step))
    
    if len(k_range) < 3:
        # Quá ít điểm để vẽ elbow
        return k_min, k_range, [], None
    
    wcss_list = []
    total_steps = len(k_range)
    for i, k in enumerate(k_range):
        if progress_callback:
            progress_callback(k, i + 1, total_steps)
            
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        kmeans.fit(X_pca)
        wcss_list.append(kmeans.inertia_)
    
    # ---- Phát hiện Elbow bằng Maximum Curvature ----
    # Chuẩn hóa trục về [0, 1] để curvature không bị bias bởi scale
    x = np.array(k_range, dtype=float)
    y = np.array(wcss_list, dtype=float)
    
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    
    # Tính khoảng cách từ mỗi điểm tới đường thẳng nối đầu-cuối
    # (Phương pháp "Maximum Distance" - ổn định hơn đạo hàm bậc 2)
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        k_optimal = k_range[len(k_range) // 2]
    else:
        line_unit = line_vec / line_len
        distances = []
        for i in range(len(k_range)):
            point = np.array([x_norm[i], y_norm[i]])
            # Khoảng cách vuông góc từ điểm tới đường thẳng
            proj = np.dot(point - p1, line_unit)
            closest_on_line = p1 + proj * line_unit
            dist = np.linalg.norm(point - closest_on_line)
            distances.append(dist)
        
        elbow_idx = np.argmax(distances)
        k_optimal = k_range[elbow_idx]
        
    # 1. Mốc Tiêu Chuẩn (🔴)
    k_standard = k_optimal
    
    # 2. Mốc Tiết Kiệm (🟡) - Điểm tốt nhất (Score chìm sâu nhất) trước mốc Tối ưu
    if elbow_idx > 0:
        subset_scores = wcss_list[:elbow_idx]
        best_eco_idx = int(np.argmin(subset_scores))
        k_economy = k_range[best_eco_idx]
    else:
        k_economy = k_range[0]
            
    # 3. Loại bỏ Mốc Dư Dả (Luxury Drop)
    # 4. Tính toán Cường độ Hao hụt (Loss Metric)
    std_idx = k_range.index(k_standard)
    eco_idx = k_range.index(k_economy)
    val_std = wcss_list[std_idx]
    val_eco = wcss_list[eco_idx]
    
    # Loss metric tỷ lệ phần trăm (do wcss_eco > wcss_std, chênh lệch dương)
    loss_pct = 0.0
    if val_std > 0:
        loss_pct = round(((val_eco - val_std) / val_std) * 100, 1)

    k_dict = {
        'standard': k_standard,
        'economy': k_economy,
        'economy_loss_pct': loss_pct
    }
    
    return k_dict, k_range, wcss_list, None


# ==============================================================================
# HÀM PHỤ TRỢ (TRỰC QUAN HÓA XUẤT CHO UI)
# ==============================================================================
def plot_saturation_curve(index_range, scores, val_dict, is_clhs=False):
    """
    Kết xuất Đồ thị hàm Nội suy mục tiêu và trích xuất hiển thị các Mốc Tối Ưu / Tiết Kiệm (Nếu có).
    Hàm này được tách nhỏ để Giao diện UI có thể Live Rendering liên tục.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    if is_clhs:
        ax.plot(index_range, scores, 'g-o', markersize=5, linewidth=2, label='cLHS Objective Score')
        ax.set_xlabel('Số lượng mẫu (N)', fontsize=11)
        ax.set_ylabel('Hàm mục tiêu O (Score)', fontsize=11)
        ax.set_title('Đường Bão hòa cLHS (Saturation Point)', fontsize=13, fontweight='bold')
    else:
        ax.plot(index_range, scores, 'b-o', markersize=5, linewidth=2, label='WCSS (Inertia)')
        ax.set_xlabel('Số cụm K (Số điểm lấy mẫu)', fontsize=11)
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=11)
        ax.set_title('Biểu đồ Cùi Chỏ (Elbow Method) — Tìm K tối ưu', fontsize=13, fontweight='bold')
        
    try:
        # Plot Standard
        std_k = val_dict.get('standard')
        if std_k is not None and std_k in index_range:
            std_idx = index_range.index(std_k)
            ax.axvline(x=std_k, color='red', linestyle='--', linewidth=2, label=f"Tối ưu = {std_k}")
            ax.scatter([std_k], [scores[std_idx]], color='red', s=200, zorder=5, edgecolors='black')
        
        # Economy plotting removed as per user request
    except Exception as e:
        print(f"Plot Error: {e}")
        pass
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig


def snap_to_inner_boundary(sample_points, coords_valid, boundary_path, target_crs, inward_buffer_m=-30.0):
    """
    Hậu xử lý Không gian (Spatial Post-Processing): 
    Kiểm tra các điểm mẫu. Nếu điểm nằm ngoài vùng lõi an toàn (vùng đệm thụt vào -30m), 
    chiếu vuông góc điểm đó vào ranh giới an toàn, sau đó snap nó vào pixel thực tế gần nhất trong coords_valid.
    """
    import os
    import geopandas as gpd
    from shapely.geometry import Point
    import numpy as np
    from scipy.spatial import cKDTree
    from shapely.ops import nearest_points
    
    if not boundary_path or not os.path.exists(boundary_path):
        return sample_points
        
    try:
        gdf = gpd.read_file(boundary_path)
        if not gdf.crs:
            return sample_points
            
        gdf = gdf.to_crs(target_crs)
        is_geo = gdf.crs.is_geographic
        buffer_deg = inward_buffer_m / 111320.0 if is_geo else inward_buffer_m
        
        gdf_inner = gdf.buffer(buffer_deg)
        gdf_inner = gdf_inner[~gdf_inner.is_empty]
        
        if gdf_inner.empty:
            return sample_points
            
        unified_inner = gdf_inner.unary_union
        pts = gpd.GeoSeries([Point(x, y) for x, y in sample_points], crs=target_crs)
        is_inside = pts.intersects(unified_inner).values
        
        snapped_points = []
        for i, pt_is_in in enumerate(is_inside):
            pt = pts[i]
            if pt_is_in:
                snapped_points.append(sample_points[i])
            else:
                p1, p2 = nearest_points(pt, unified_inner)
                snapped_points.append(np.array([p2.x, p2.y]))
                
        tree_valid = cKDTree(coords_valid)
        _, valid_idx = tree_valid.query(np.array(snapped_points))
        final_points = coords_valid[valid_idx]
        
        return final_points
    except Exception as e:
        print(f"Lỗi khi Snap điểm vào ranh giới lõi: {e}")
        return sample_points


def k_means_sampling(X_pca, coords_valid, num_samples, src_crs=None, d_max=None, add_short_lags=False, boundary_path=None, progress_callback=None):
    """K-Means trên PC Space kết hợp Vá lỗ hổng không gian và Cặp cự ly gần (Short-lag pairs)."""
    if progress_callback: progress_callback("1️⃣ Chạy K-Means phân tầng Không gian thuộc tính...")
    kmeans = KMeans(n_clusters=num_samples, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    
    centers = kmeans.cluster_centers_
    sample_points = []
    
    for i, center in enumerate(centers):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) == 0:
            continue
            
        cluster_points = X_pca[cluster_indices]
        distances_to_center = np.linalg.norm(cluster_points - center, axis=1)
        best_idx_in_cluster = np.argmin(distances_to_center)
        original_idx = cluster_indices[best_idx_in_cluster]
        sample_points.append(coords_valid[original_idx])
        
    sample_points = np.array(sample_points)
    
    # 2. Spatial Infilling (Vá lỗ hổng địa lý)
    if d_max is not None and d_max:
        # Tính toán D_max động học dựa trên số lượng N hiện tại (The N Paradox)
        x_min, y_min = np.min(coords_valid, axis=0)
        x_max, y_max = np.max(coords_valid, axis=0)
        area_m2 = (x_max - x_min) * (y_max - y_min)
        num_samples_safe = max(1, num_samples)
        d_grid = np.sqrt(area_m2 / num_samples_safe)
        dynamic_dmax = max(100.0, d_grid * 1.5)
        
        if progress_callback: progress_callback(f"2️⃣ Vá lỗ hổng D_max Tự động ({dynamic_dmax:,.0f}m) rải theo N={num_samples}...")
        from scipy.spatial import cKDTree
        
        infill_points = []
        max_infill = int(num_samples * 0.5) # Giới hạn số điểm rải bù max 50%
        
        # Subsample để tối ưu tốc độ nếu bản đồ quá lớn
        eval_coords = coords_valid
        if len(coords_valid) > 50000:
            np.random.seed(42)
            eval_coords = coords_valid[np.random.choice(len(coords_valid), 50000, replace=False)]
            
        current_samples = sample_points.copy()
        for step in range(max_infill):
            tree = cKDTree(current_samples)
            distances, _ = tree.query(eval_coords)
            max_dist_idx = np.argmax(distances)
            
            if distances[max_dist_idx] > dynamic_dmax:
                new_point = eval_coords[max_dist_idx]
                infill_points.append(new_point)
                current_samples = np.vstack([current_samples, new_point])
            else:
                break
                
        if len(infill_points) > 0:
            sample_points = np.vstack([sample_points, infill_points])
            if progress_callback: progress_callback(f"✅ Đã chèn bù {len(infill_points)} điểm vào các vùng trống.")
            
    # 3. Short-lag Pairs (Bắt Nugget)
    if add_short_lags:
        if progress_callback: progress_callback("3️⃣ Tạo Cặp điểm cự ly gần (Short-lags) để ổn định Variogram...")
        from scipy.spatial import cKDTree
        
        n_pairs = max(1, int(len(sample_points) * 0.10)) # Lấy 10% làm điểm gốc
        np.random.seed(42)
        indices = np.random.choice(len(sample_points), n_pairs, replace=False)
        
        short_lag_points = []
        tree_valid = cKDTree(coords_valid)
        
        # --- Tính toán Quy mô Động học dựa trên Bounding Box Không gian ---
        x_min, y_min = np.min(coords_valid, axis=0)
        x_max, y_max = np.max(coords_valid, axis=0)
        diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        
        # Thiết lập cửa sổ nội suy cự ly ngắn (Short-lags)
        # Min lag: Ít nhất 100m, nới theo 1% quy mô vùng, nhưng tuyệt đối không quá 300m
        min_lag = min(max(100.0, diagonal * 0.01), 300.0)
        # Max lag: Khống chế 5% độ sải cánh, nhưng trần tối đa là 1500m
        max_lag = min(diagonal * 0.05, 1500.0)
        
        if progress_callback: progress_callback(f"📏 Tầm bao phủ cực đại: {diagonal/1000:.1f} km. Rải biên độ Lag động học: {min_lag:.0f}m ➝ {max_lag:.0f}m")
        
        for idx in indices:
            base_pt = sample_points[idx]
            # Bắn cặp điểm ra ngẫu nhiên theo phân phối nới lỏng động học
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(min_lag, max_lag)
            offset = np.array([dist * np.cos(angle), dist * np.sin(angle)])
            target_pt = base_pt + offset
            
            # Dịch chuyển vào pixel hợp lệ (trong ranh giới) gần nhất
            _, nearest_idx = tree_valid.query(target_pt)
            short_lag_points.append(coords_valid[nearest_idx])
            
        if len(short_lag_points) > 0:
            sample_points = np.vstack([sample_points, short_lag_points])
            if progress_callback: progress_callback(f"✅ Đã phân bổ thêm {len(short_lag_points)} điểm bắt Nugget.")

    if boundary_path and os.path.exists(boundary_path):
        if progress_callback: progress_callback("🚀 Đang nắn chỉnh tọa độ lọt lòng an toàn (Post-Processing)...")
        target_crs = src_crs if src_crs else "EPSG:32648"
        sample_points = snap_to_inner_boundary(sample_points, coords_valid, boundary_path, target_crs)

    # Đóng gói kết quả
    df_samples = pd.DataFrame(sample_points, columns=['X_UTM', 'Y_UTM'])
    df_samples['Point_ID'] = [f"P{i+1}" for i in range(len(df_samples))]
    
    if src_crs is not None:
        try:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame(df_samples, geometry=gpd.points_from_xy(df_samples.X_UTM, df_samples.Y_UTM), crs=src_crs)
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
            df_samples['Longitude'] = gdf_wgs84.geometry.x
            df_samples['Latitude'] = gdf_wgs84.geometry.y
            cols = ['Point_ID', 'Longitude', 'Latitude', 'X_UTM', 'Y_UTM']
            df_samples = df_samples[cols]
        except Exception as e:
            print("Lỗi chuyển toạ độ:", e)
            
    return df_samples


def clhs_objective_analysis(X_pca, X_cat=None, n_min=10, n_max=100, n_step=5, progress_callback=None):
    """Quét Hàm mục tiêu để tìm Điểm bão hòa cLHS."""
    import matplotlib.pyplot as plt
    from scipy.stats.qmc import LatinHypercube
    from scipy.spatial import cKDTree
    
    n_range = list(range(n_min, min(n_max + 1, len(X_pca)), n_step))
    if len(n_range) < 3:
        return n_min, n_range, [], None
        
    corr_orig = np.corrcoef(X_pca, rowvar=False) if X_pca.shape[1] > 1 else np.array([[1]])
    
    eval_pca = X_pca
    eval_cat = X_cat
    if len(X_pca) > 20000:
        np.random.seed(42)
        idx_choice = np.random.choice(len(X_pca), 20000, replace=False)
        eval_pca = X_pca[idx_choice]
        if X_cat is not None:
            eval_cat = X_cat[idx_choice]
    tree_pca = cKDTree(eval_pca)
    
    scores = []
    total_steps = len(n_range)
    for i, n in enumerate(n_range):
        if progress_callback:
            progress_callback(n, i + 1, total_steps)
            
        engine = LatinHypercube(d=X_pca.shape[1], seed=42)
        lhs_sample = engine.random(n=n)
        
        target_pca = np.empty_like(lhs_sample)
        for col in range(X_pca.shape[1]):
            target_pca[:, col] = np.quantile(eval_pca[:, col], lhs_sample[:, col])
        
        _, best_indices = tree_pca.query(target_pca)
        best_indices = list(set(best_indices))
        while len(best_indices) < n:
            r = np.random.randint(0, len(eval_pca))
            if r not in best_indices: best_indices.append(r)
        best_indices = np.array(best_indices[:n])
        
        def calc_obj(subset_idx):
            X_sub = eval_pca[subset_idx]
            corr_sub = np.corrcoef(X_sub, rowvar=False) if X_sub.shape[1] > 1 else np.array([[1]])
            corr_diff = np.sum(np.abs(corr_orig - corr_sub))
            from scipy.stats import wasserstein_distance
            wd_diff = sum(wasserstein_distance(eval_pca[:, c], X_sub[:, c]) for c in range(eval_pca.shape[1]))
            
            O_cat = 0
            if eval_cat is not None:
                C_sub = eval_cat[subset_idx]
                for c in range(eval_cat.shape[1]):
                    # Match histogram proportions
                    pop_counts = np.bincount(eval_cat[:, c].astype(int))
                    pop_prop = pop_counts / len(eval_cat)
                    
                    sub_counts = np.bincount(C_sub[:, c].astype(int), minlength=len(pop_counts))
                    sub_prop = sub_counts / len(C_sub)
                    
                    # Weight categorical penalty heavily so it distributes well
                    O_cat += np.sum(np.abs(pop_prop - sub_prop)) * len(eval_pca[0])
                    
            return corr_diff + wd_diff + O_cat
            
        current_score = calc_obj(best_indices)
        
        # Fast Annealing (50 iters) for curve scanning
        T = 1.0
        for _ in range(50):
            swap_out = np.random.randint(0, n)
            swap_in = np.random.randint(0, len(eval_pca))
            new_idx = best_indices.copy()
            new_idx[swap_out] = swap_in
            new_score = calc_obj(new_idx)
            if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
                best_indices = new_idx
                current_score = new_score
            T *= 0.9
            
        scores.append(current_score)
        
    x = np.array(n_range, dtype=float)
    y = np.array(scores, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        n_optimal = n_range[len(n_range) // 2]
    else:
        line_unit = line_vec / line_len
        distances = []
        for i in range(len(n_range)):
            point = np.array([x_norm[i], y_norm[i]])
            proj = np.dot(point - p1, line_unit)
            closest_on_line = p1 + proj * line_unit
            dist = np.linalg.norm(point - closest_on_line)
            distances.append(dist)
        
        elbow_idx = np.argmax(distances)
        n_optimal = n_range[elbow_idx]
        
    # 1. Mốc Tiêu Chuẩn (🔴)
    n_standard = n_optimal
    
    # 2. Mốc Tiết Kiệm (🟡) - Điểm tốt nhất (Score chìm sâu nhất) trước mốc Tối ưu
    if elbow_idx > 0:
        subset_scores = scores[:elbow_idx]
        best_eco_idx = int(np.argmin(subset_scores))
        n_economy = n_range[best_eco_idx]
    else:
        n_economy = n_range[0]
            
    # 3. Loại bỏ Mốc Dư Dả (Luxury Drop)
    # 4. Tính toán Cường độ Hao hụt (Loss Metric)
    std_idx = n_range.index(n_standard)
    eco_idx = n_range.index(n_economy)
    val_std = scores[std_idx]
    val_eco = scores[eco_idx]
    
    # Loss metric tỷ lệ phần trăm (do val_eco > val_std)
    loss_pct = 0.0
    if val_std > 0:
        loss_pct = round(((val_eco - val_std) / val_std) * 100, 1)
            
    n_dict = {
        'standard': n_standard,
        'economy': n_economy,
        'economy_loss_pct': loss_pct
    }
        
    return n_dict, n_range, scores, None


def clhs_sampling(X_pca, coords_valid, num_samples, X_cat=None, src_crs=None, d_max=None, add_short_lags=False, boundary_path=None, progress_callback=None):
    """Fast Empirical cLHS Sampling tích hợp Không gian."""
    from scipy.stats.qmc import LatinHypercube
    from scipy.spatial import cKDTree
    
    if progress_callback: progress_callback("1️⃣ Sinh khối Siêu lập phương Latin (LHS)...")
    
    engine = LatinHypercube(d=X_pca.shape[1], seed=42)
    lhs_sample = engine.random(n=num_samples)
    
    if progress_callback: progress_callback("Đang đồng bộ Xác suất cLHS lên Vệ tinh...")
    target_pca = np.empty_like(lhs_sample)
    
    eval_pca = X_pca
    eval_coords = coords_valid
    
    for col in range(X_pca.shape[1]):
        target_pca[:, col] = np.quantile(eval_pca[:, col], lhs_sample[:, col])
        
    tree_pca = cKDTree(eval_pca)
    _, best_indices = tree_pca.query(target_pca)
    best_indices = list(set(best_indices))
    while len(best_indices) < num_samples:
        r = np.random.randint(0, len(eval_pca))
        if r not in best_indices: best_indices.append(r)
    best_indices = np.array(best_indices[:num_samples])
    
    if progress_callback: progress_callback("Đang Tối ưu hóa Simulated Annealing (Đồng bộ Địa hình & Nhóm đất)...")
    corr_orig = np.corrcoef(eval_pca, rowvar=False) if eval_pca.shape[1] > 1 else np.array([[1]])
    
    # Tiền xử lý (Precompute) để tăng tốc độ SA Loop
    target_pca_dist = []
    if len(eval_pca) > 100000:
        np.random.seed(42)
        idx_dist = np.random.choice(len(eval_pca), 100000, replace=False)
        for c in range(eval_pca.shape[1]):
            target_pca_dist.append(eval_pca[idx_dist, c])
    else:
        for c in range(eval_pca.shape[1]):
            target_pca_dist.append(eval_pca[:, c])
            
    pop_props = []
    if X_cat is not None:
        for c in range(X_cat.shape[1]):
            pop_counts = np.bincount(X_cat[:, c].astype(int))
            pop_props.append(pop_counts / len(X_cat))
    
    from scipy.stats import wasserstein_distance
    def calc_obj(subset_idx):
        X_sub = eval_pca[subset_idx]
        corr_sub = np.corrcoef(X_sub, rowvar=False) if X_sub.shape[1] > 1 else np.array([[1]])
        wd_diff = sum(wasserstein_distance(target_pca_dist[c], X_sub[:, c]) for c in range(eval_pca.shape[1]))
        O_cont = np.sum(np.abs(corr_orig - corr_sub)) + wd_diff
        
        O_cat = 0
        if X_cat is not None:
            C_sub = X_cat[subset_idx]
            for c in range(X_cat.shape[1]):
                sub_counts = np.bincount(C_sub[:, c].astype(int), minlength=len(pop_props[c]))
                sub_prop = sub_counts / len(C_sub)
                O_cat += np.sum(np.abs(pop_props[c] - sub_prop)) * len(eval_pca[0])
                
        return O_cont + O_cat
        
    current_score = calc_obj(best_indices)
    T = 1.0
    for _ in range(1000):
        swap_out = np.random.randint(0, num_samples)
        swap_in = np.random.randint(0, len(eval_pca))
        new_idx = best_indices.copy()
        new_idx[swap_out] = swap_in
        new_score = calc_obj(new_idx)
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
            best_indices = new_idx
            current_score = new_score
        T *= 0.99

    sample_points = eval_coords[best_indices]

    if d_max is not None and d_max:
        is_geo = False
        if src_crs is not None:
            try:
                import pyproj
                is_geo = pyproj.CRS.from_user_input(src_crs).is_geographic
            except: pass
            
        x_min, y_min = np.min(coords_valid, axis=0)
        x_max, y_max = np.max(coords_valid, axis=0)
        area_m2 = (x_max - x_min) * (y_max - y_min)
        num_samples_safe = max(1, num_samples)
        d_grid = np.sqrt(area_m2 / num_samples_safe)
        
        min_limit = 100.0 / 111320.0 if is_geo else 100.0
        dynamic_dmax = max(min_limit, d_grid * 1.5)
        
        if progress_callback: progress_callback(f"2️⃣ Vá lỗ hổng D_max Tự động ({dynamic_dmax:,.0f}m) rải theo N={num_samples}...")
        infill_points = []
        max_infill = int(num_samples * 0.5)
        eval_coords_spatial = coords_valid
        if len(coords_valid) > 50000:
            np.random.seed(42)
            eval_coords_spatial = coords_valid[np.random.choice(len(coords_valid), 50000, replace=False)]
            
        current_samples = sample_points.copy()
        for step in range(max_infill):
            tree = cKDTree(current_samples)
            distances, _ = tree.query(eval_coords_spatial)
            max_dist_idx = np.argmax(distances)
            if distances[max_dist_idx] > dynamic_dmax:
                new_point = eval_coords_spatial[max_dist_idx]
                infill_points.append(new_point)
                current_samples = np.vstack([current_samples, new_point])
            else:
                break
        if len(infill_points) > 0:
            sample_points = np.vstack([sample_points, infill_points])
            if progress_callback: progress_callback(f"✅ Đã chèn bù {len(infill_points)} điểm vào các vùng trống.")
            
    if add_short_lags:
        if progress_callback: progress_callback("3️⃣ Tạo Cặp điểm cự ly gần (Short-lags) bắt Nugget Effect...")
        n_pairs = max(1, int(len(sample_points) * 0.10))
        np.random.seed(42)
        indices = np.random.choice(len(sample_points), n_pairs, replace=False)
        short_lag_points = []
        tree_valid = cKDTree(coords_valid)
        
        is_geo = False
        if src_crs is not None:
            try:
                import pyproj
                is_geo = pyproj.CRS.from_user_input(src_crs).is_geographic
            except: pass
            
        # Geostatistical standard for Composite Sampling (Lấy mẫu gộp/trộn):
        # Vì thực tế nông nghiệp lấy 5-10 lõi đất trong bán kính 30-50m để trộn chung (Composite sample),
        # kích thước hỗ trợ (Support Size) đã là ~50m. Do đó khoảng cách 2 điểm (lag) BẮT BUỘC phải > 50m
        # để tránh trùng lặp mẫu. Con số 100m là mức tối thiểu cực kỳ chuẩn xác!
        min_lag = 100.0 / 111320.0 if is_geo else 100.0
        max_lag = 300.0 / 111320.0 if is_geo else 300.0
        
        if progress_callback: progress_callback(f"📏 Rải biên độ Lag vi mô (Composite Support): 100m ➝ 300m")

        for idx in indices:
            base_pt = sample_points[idx]
            # Bắt góc bắn radar và cự ly bắn
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(min_lag, max_lag)
            offset = np.array([dist * np.cos(angle), dist * np.sin(angle)])
            target_pt = base_pt + offset
            _, nearest_idx = tree_valid.query(target_pt)
            short_lag_points.append(coords_valid[nearest_idx])
        if len(short_lag_points) > 0:
            sample_points = np.vstack([sample_points, short_lag_points])
            if progress_callback: progress_callback(f"✅ Đã phân bổ thêm {len(short_lag_points)} điểm bắt Nugget.")

    if boundary_path and os.path.exists(boundary_path):
        if progress_callback: progress_callback("🚀 Đang nắn chỉnh tọa độ lọt lòng an toàn (Post-Processing)...")
        target_crs = src_crs if src_crs else "EPSG:32648"
        sample_points = snap_to_inner_boundary(sample_points, coords_valid, boundary_path, target_crs)

    import pandas as pd
    df_samples = pd.DataFrame(sample_points, columns=['X_UTM', 'Y_UTM'])
    df_samples['Point_ID'] = [f"P{i+1}" for i in range(len(df_samples))]
    if src_crs is not None:
        try:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame(df_samples, geometry=gpd.points_from_xy(df_samples.X_UTM, df_samples.Y_UTM), crs=src_crs)
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
            df_samples['Longitude'] = gdf_wgs84.geometry.x
            df_samples['Latitude'] = gdf_wgs84.geometry.y
            cols = ['Point_ID', 'Longitude', 'Latitude', 'X_UTM', 'Y_UTM']
            df_samples = df_samples[cols]
        except Exception:
            pass
    return df_samples
