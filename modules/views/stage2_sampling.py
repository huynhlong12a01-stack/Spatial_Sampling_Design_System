import streamlit as st
import os
import glob
import json
import numpy as np
import joblib
from modules.sampling_opt import (
    stack_rasters, elbow_analysis, k_means_sampling, 
    clhs_objective_analysis, clhs_sampling
)


def _calculate_absolute_min(project_dir, n_pcs, model_type="MLR"):
    """Tính số mẫu TỐI THIỂU cho phương pháp, tự scale theo diện tích.
    
    Vùng nhỏ (<500 ha): giữ ngưỡng thấp, ưu tiên tiết kiệm
    Vùng lớn (>1000 ha): tăng dần, vì variogram cần nhiều lag pairs hơn
    """
    import geopandas as gpd
    # Chúng ta phải đọc diện tích của Ranh giới ruộng mía, không phải ROI (ROI là vùng chữ nhật xin ảnh vệ tinh)
    boundary_path = os.path.join(project_dir, 'sugarcane_boundary.geojson')
    area_ha = 0
    if os.path.exists(boundary_path):
        gdf = gpd.read_file(boundary_path)
        if gdf.crs and gdf.crs.is_geographic:
            config_path = os.path.join(project_dir, 'stage1_config.json')
            target_crs = "EPSG:32648"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    conf = json.load(f)
                    target_crs = conf.get('crs', target_crs)
            gdf_proj = gdf.to_crs(target_crs)
        else:
            gdf_proj = gdf
        area_ha = gdf_proj.geometry.area.sum() / 10_000
    
    # ---- Variogram: scale theo diện tích ----
    # Chuẩn Khoa học Geostatistics (Webster & Oliver, 2007): Cần tối thiểu 50-75 điểm 
    # để đảm bảo Semivariogram có đủ số lượng point pairs ở mỗi lag bin.
    import numpy as np
    if area_ha <= 100:
        variogram_min = 50
    elif area_ha <= 500:
        variogram_min = 50 + int(20 * np.log10(area_ha / 100 + 1))
    else:
        variogram_min = 75 + int(30 * np.log10(area_ha / 500 + 1))
    
    # ---- Regression: phụ thuộc phương pháp ----
    if model_type == "MLR":
        regression_min = n_pcs * 5
    else:
        regression_min = max(n_pcs * 8, 50)
    
    # ---- Phủ không gian: 1.5×√ha ----
    spatial_min = max(20, int(np.ceil(1.5 * np.sqrt(area_ha)))) if area_ha > 0 else 20
    
    absolute_min = max(variogram_min, regression_min, spatial_min)
    
    dominant = []
    if absolute_min == variogram_min:
        dominant.append(f"Variogram cần ≥ {variogram_min} điểm (diện tích {area_ha:.0f} ha)")
    if absolute_min == regression_min:
        if model_type == "MLR":
            dominant.append(f"MLR ({n_pcs} PCs × 5) cần ≥ {regression_min} điểm")
        else:
            dominant.append(f"RF ({n_pcs} PCs × 8) cần ≥ {regression_min} điểm")
    if absolute_min == spatial_min:
        dominant.append(f"Phủ {area_ha:.0f} ha cần ≥ {spatial_min} điểm")
    
    return absolute_min, area_ha, dominant


def _save_stage2_config(project_dir, data):
    config_path = os.path.join(project_dir, 'stage2_config.json')
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)


def _load_stage2_config(project_dir):
    config_path = os.path.join(project_dir, 'stage2_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def _restore_session_from_disk(project_dir):
    cfg = _load_stage2_config(project_dir)
    if cfg is None:
        return False
    st.session_state['s2_k_optimal'] = cfg['k_optimal']
    st.session_state['s2_min_mlr'] = cfg['min_mlr']
    st.session_state['s2_min_rf'] = cfg['min_rf']
    st.session_state['s2_n_pcs'] = cfg['n_pcs']
    st.session_state['s2_area_ha'] = cfg['area_ha']
    
    if 'ui_state' in cfg:
        ui = cfg['ui_state']
        if 'algo_choice' in ui: st.session_state['s2_algo_choice'] = ui['algo_choice']
        if 'k_min' in ui: st.session_state['s2_k_min_val'] = ui['k_min']
        if 'k_max' in ui: st.session_state['s2_k_max_val'] = ui['k_max']
        if 'k_step' in ui: st.session_state['s2_k_step_val'] = ui['k_step']
        if 'd_max' in ui: st.session_state['s2_d_max_val'] = ui['d_max']
        if 'add_lags' in ui: st.session_state['s2_add_lags_val'] = ui['add_lags']
        
    elbow_path = os.path.join(project_dir, 'stage2_elbow.png')
    if os.path.exists(elbow_path):
        st.session_state['s2_elbow_img_path'] = elbow_path
    return True


def render_stage2():
    st.header("Giai đoạn 2: Thiết kế Không gian Mẫu (Sampling Design)")
    project_dir = st.session_state['project_dir']
    cov_dir = os.path.join(project_dir, 'covariates')
    spl_dir = os.path.join(project_dir, 'samples')
    
    raster_files = glob.glob(os.path.join(cov_dir, '*.tif'))
    if len(raster_files) == 0:
        st.error("Dự án chưa có file `.tif` Covariates nào. Hãy quay lại Giai đoạn 1.")
        return
        
    st.info(f"**{len(raster_files)}** lớp Covariates: " + ", ".join([os.path.basename(f) for f in raster_files]))
    
    # Khôi phục từ ổ cứng
    if 's2_k_optimal' not in st.session_state:
        _restore_session_from_disk(project_dir)
    
    # =====================================================
    # A. CHỌN PHƯƠNG PHÁP
    # =====================================================
    st.subheader("A. Phương pháp Hồi quy cho Giai đoạn 3")
    
    saved_cfg = _load_stage2_config(project_dir)
    saved_model = saved_cfg.get('model_type', 'MLR') if saved_cfg else 'MLR'
    default_radio = 0 if saved_model == 'MLR' else 1
    
    model_choice = st.radio(
        "Chọn phương pháp sẽ dùng ở Giai đoạn 3:",
        ["MLR (Multiple Linear Regression)", "RF (Random Forest)"],
        index=default_radio,
        captions=[
            "Cần ÍT mẫu hơn. Phù hợp khi ngân sách hạn chế. Yêu cầu PCA bắt buộc.",
            "Cần NHIỀU mẫu hơn. Bắt lấy quan hệ phi tuyến tốt hơn. PCA tùy chọn."
        ]
    )
    model_type = "MLR" if "MLR" in model_choice else "RF"
    
    # =====================================================
    # B. CHẠY PHÂN TÍCH + RẢI ĐIỂM (1 nút duy nhất)
    # =====================================================
    st.markdown("---")
    st.subheader("B. Phân tích & Rải điểm tự động")
    
    pca_dir = os.path.join(project_dir, 'pca')
    pca_model_path = os.path.join(pca_dir, 'pca_model.joblib')
    pc_rasters = sorted(glob.glob(os.path.join(pca_dir, 'PC*.tif'))) if os.path.exists(pca_dir) else []
    
    if not os.path.exists(pca_model_path) or len(pc_rasters) == 0:
        st.warning("⚠️ PCA chưa được tính ở Giai đoạn 1. Hãy quay lại chạy Giai đoạn 1 trước.")
        return
    
    pca_pack = joblib.load(pca_model_path)
    n_pcs = pca_pack.get('n_components', len(pc_rasters))
    explained = pca_pack.get('explained_variance', 0)
    
    st.success(f"🧬 **PCA từ Giai đoạn 1:** {pca_pack.get('original_features', len(pc_rasters))} biến liên tục → **{n_pcs} PC** ({explained:.1f}% variance)")
    
    # Xử lý thông số dựa trên chuẩn
    min_mlr_default, area_ha_val, _ = _calculate_absolute_min(project_dir, n_pcs, "MLR")
    min_rf_default, _, _ = _calculate_absolute_min(project_dir, n_pcs, "RF")
    abs_min_default = min_mlr_default if model_type == "MLR" else min_rf_default
    
    st.write(f"- 📏 Diện tích: **{area_ha_val:,.0f} ha** | Tối thiểu MLR: {min_mlr_default} | Tối thiểu RF: {min_rf_default}")

    # Chọn thuật toán
    st.markdown("---")
    st.subheader("B. Lựa chọn Thuật toán Lấy mẫu")
    algo_options = ["K-Means Clustering", "cLHS (Mô hình Siêu lập phương)"]
    default_algo_idx = 0
    if st.session_state.get('s2_algo_choice') in algo_options:
        default_algo_idx = algo_options.index(st.session_state['s2_algo_choice'])

    algo_choice = st.radio(
        "Thuật toán cốt lõi:",
        algo_options,
        index=default_algo_idx,
        key='s2_algo_choice',
        captions=["Tốt cho khu vực nhỏ, bắt dị thường.", "Chuyên dụng cho tỷ lệ lớn (Vùng/Huyện), mô phỏng hoàn hảo phân phối xác suất."]
    )
    is_clhs = "cLHS" in algo_choice

    with st.expander("📖 Hướng dẫn Dành cho Người mới: Rào chắn Không gian là gì?"):
        st.markdown('''
        **Khái niệm cơ bản:** Cả **K-Means** và **cLHS** đều là các thuật toán "Lọc theo độ đa dạng chất đất". Chúng nhìn ảnh vệ tinh để gom cụm: chỗ nào đất đỏ, chỗ nào đất cát, rồi mới nhả mẫu đại diện vào đó, thay vì cứ rải lưới caro vung tiền vô tội vạ. Tức là chúng nội suy bằng **Đặc tính vệ tinh**.

        Tuy nhiên, do quá tập trung nhìn "Chất đất", thỉnh thoảng máy tính hay bị **mù về Khoảng cách địa lý**. Vì thế chúng tôi gài thêm 2 lá chắn "Không gian" (Spatial Optimization) chạy ngầm phía sau cùng để bịt các lỗ hổng toán học:

        **1. Bán kính Vá lỗ hổng không gian ($D_{max}$):**
        - *Ví dụ:* Có một xã Nông nghiệp cực rộng (2,000 ha) có tính chất đất y hệt xã lân cận. Do thấy "quá giống nhau", thuật toán sẽ bỏ đi nơi khác lấy mẫu để tiết kiệm điểm, kết quả là... **bỏ trống trắng tinh** nguyên một xã này không bốc mẫu nào. Khi vẽ bản đồ Kriging, khoảng cách quá xa sẽ khiến vùng trống bị sai số nội suy khủng khiếp (Phương sai bay lơ lửng).
        - *Giải pháp:* Nếu bạn đặt $D_{max} = 1000m$. Máy sẽ kích hoạt Radar dò mìn quét qua toàn bản đồ, hễ cứ phát hiện lọt thỏm một khoảng đất trống $>1km$ mà không có dòng tọa độ nào, nó sẽ gắt gao tự sinh thêm 1 điểm mẫu (Infilling) ném vào giữa vùng lọt thỏm đó để trám vá lại.
        
        **2. Khởi tạo Cặp điểm song sinh (Short-lags):**
        - *Ví dụ:* Để máy tính vẽ được đường dốc Semivariogram (Biểu đồ phương sai lõi của Kriging), nó yêu cầu phải có dữ liệu để tính *"Đất thay đổi như thế nào nếu tôi đi bước ngắn 30 mét"*. Nếu các điểm mẫu cấp tỉnh toàn cách xa nhau 5 Kilomet, máy tính đành chịu thua ở cự ly gần và sinh ra Lỗi Nugget!
        - *Giải pháp:* Nếu tích chọn, máy sẽ tự bốc thăm ngẫu nhiên 10% các điểm, tạo bóng nhân bản của chúng, rồi ném rớt ngay vị trí cự ly siêu gần (cách thân cây gốc tâm 30-50m). Điều kiện có những "Cặp anh em song sinh" này sẽ hóa giải hoàn toàn điểm mù cự ly gần của máy.
        ''')

    # Cài đặt tùy chỉnh (Expander)
    with st.expander(f"⚙️ Tùy chỉnh tham số {'cLHS' if is_clhs else 'K-Means'} & Không gian", expanded=False):
        st.markdown(f"**1. Cài đặt Quét Bão hòa {'(Objective Score)' if is_clhs else '(Elbow Range)'}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            loaded_k_min = st.session_state.get('s2_k_min_val', 10)
            suggested_k_min = int(max(loaded_k_min, abs_min_default))
            k_min_val = st.number_input("N_min (Nhỏ nhất)", min_value=5, max_value=5000, value=suggested_k_min, step=5, key='s2_k_min_val')
        with col2:
            default_k_max = int(min(max(abs_min_default * 3, k_min_val + 50), 500))
            k_max_val = st.number_input("N_max (Lớn nhất)", min_value=int(k_min_val + 5), max_value=10000, value=st.session_state.get('s2_k_max_val', default_k_max), step=10, key='s2_k_max_val')
        with col3:
            k_step_val = st.number_input("N_step (Bước nhảy)", min_value=1, max_value=100, value=st.session_state.get('s2_k_step_val', 5), step=1, key='s2_k_step_val')
            
        st.markdown("**2. Rào chắn Không gian (Spatial Optimization)**")
        col4, col5 = st.columns(2)
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            d_max_val = st.checkbox("Bật Rào chắn Vá Lỗ hổng D_max", value=st.session_state.get('s2_d_max_val', True), key='s2_d_max_val')
            st.caption("🤖 Máy tự động siết/mở cự ly D_max bám theo đường cong sinh điểm N.")
        with col5:
            st.markdown("<br>", unsafe_allow_html=True)
            add_lags_val = st.checkbox("Sinh Cặp điểm Cự ly gần (Short-lags)", value=st.session_state.get('s2_add_lags_val', True), key='s2_add_lags_val')
            st.caption("🤖 Phân bố ngẫu nhiên trên phân phối Động học tỷ lệ với Ranh giới tỉnh/huyện.")

    if st.button(f"🚀 Phân tích Mốc lấy mẫu {'Bão hòa (cLHS)' if is_clhs else 'Elbow (K-Means)'}", type="primary"):
        st.session_state['s2_setup'] = {
            'is_clhs': is_clhs, 'k_min_val': k_min_val, 'k_max_val': k_max_val, 'k_step_val': k_step_val,
            'd_max_val': d_max_val, 'add_lags_val': add_lags_val, 'model_type': model_type
        }
        with st.status("Đang phân tích Hàm Mục tiêu...", expanded=True) as status:
            import shutil
            st.write("🧹 0️⃣ Dọn dẹp Dữ liệu Mô hình cũ...")
            for d in ['models', 'outputs']:
                del_dir = os.path.join(project_dir, d)
                if os.path.exists(del_dir):
                    try: shutil.rmtree(del_dir)
                    except: pass
            for k in list(st.session_state.keys()):
                if k.startswith('s3_') or k.startswith('s4_') or k.startswith('s2_final') or k.startswith('s2_coord'):
                    del st.session_state[k]
                    
            try:
                cat_rasters = glob.glob(os.path.join(cov_dir, '*Soil_Class*.tif')) + glob.glob(os.path.join(cov_dir, '*Categorical*.tif'))
                all_rasters = pc_rasters + cat_rasters
                boundary_path = os.path.join(project_dir, 'sugarcane_boundary.geojson')
                X_pca, X_cat, coords_valid, meta, valid_mask, _, _ = stack_rasters(all_rasters)
                st.session_state['s2_X_pca'] = X_pca
                st.session_state['s2_X_cat'] = X_cat
                st.session_state['s2_coords_valid'] = coords_valid
                st.session_state['s2_meta'] = meta
                
                if is_clhs:
                    st.write(f"2️⃣ Phân tích Điểm Bão hòa cLHS (N = {k_min_val} → {k_max_val})...")
                else:
                    st.write(f"2️⃣ Quét Cùi chỏ K-Means (K = {k_min_val} → {k_max_val})...")
                    
                st.warning("🆘 **Dừng khẩn cấp:** Nếu bạn lỡ cấu hình sai, hãy nhấn nút 🛑 **Stop** nhỏ ở góc trên cùng bên phải màn hình để ngắt quá trình tính toán.")
                
                progress_bar = st.progress(0.0)
                progress_text = st.empty()
                
                def progress_update(k, current, total):
                    pct = current / total
                    progress_bar.progress(pct)
                    if is_clhs:
                        progress_text.write(f"⏳ Đang chấm điểm K-S & Tương quan... kiểm tra **N = {k}** (Bước {current} / {total})")
                    else:
                        progress_text.write(f"⏳ Đang dò tìm Elbow... kiểm tra **K = {k}** (Bước {current} / {total})")
                
                if is_clhs:
                    k_opt_dict, k_range, scores, _ = clhs_objective_analysis(
                        X_pca, X_cat=st.session_state.get('s2_X_cat'), n_min=k_min_val, n_max=k_max_val, n_step=k_step_val, progress_callback=progress_update
                    )
                else:
                    k_opt_dict, k_range, scores, _ = elbow_analysis(
                        X_pca, k_min=k_min_val, k_max=k_max_val, k_step=k_step_val, progress_callback=progress_update
                    )
                
                progress_bar.empty()
                progress_text.empty()
                
                k_optimal = k_opt_dict['standard']
                st.session_state['s2_k_dict'] = k_opt_dict
                st.session_state['s2_k_range'] = k_range
                st.session_state['s2_k_scores'] = scores
                if 's2_target_n' not in st.session_state:
                    st.session_state['s2_target_n'] = int(k_optimal)
                
                from modules.sampling_opt import plot_saturation_curve
                elbow_fig = plot_saturation_curve(k_range, scores, k_opt_dict, is_clhs=is_clhs)
                
                elbow_path = os.path.join(project_dir, 'stage2_elbow.png')
                elbow_fig.savefig(elbow_path, dpi=120, bbox_inches='tight')
                st.session_state['s2_elbow_img_path'] = elbow_path
                st.session_state['s2_analysis_done'] = True
                
                st.session_state['s2_k_optimal'] = k_optimal
                st.session_state['s2_min_mlr'] = min_mlr_default
                st.session_state['s2_min_rf'] = min_rf_default
                st.session_state['s2_n_pcs'] = n_pcs
                st.session_state['s2_area_ha'] = area_ha_val
                
                status.update(label=f"Hoàn tất Phân tích! (Tối ưu K={k_optimal})", state="complete", expanded=False)
                st.session_state['s2_final_n_est'] = None # Force compute dry-run
                st.rerun()
                
            except Exception as e:
                st.error(f"Lỗi: {e}")
                import traceback
                st.code(traceback.format_exc())
                status.update(label="Lỗi!", state="error")
                
    # =====================================================
    # INTERACTIVE B: HIỆU CHỈNH TRỰC QUAN (SAU PHÂN TÍCH)
    # =====================================================
    has_analysis = st.session_state.get('s2_analysis_done', False)
    if has_analysis:
        elbow_img_path = st.session_state.get('s2_elbow_img_path', None)
        if elbow_img_path and os.path.exists(elbow_img_path):
            st.image(elbow_img_path, width='stretch')
            
            k_opt = st.session_state['s2_k_optimal']
            k_range = st.session_state['s2_k_range']
            scores = st.session_state['s2_k_scores']
            current_target = st.session_state.get('s2_target_n', k_opt)
            k_dict = st.session_state.get('s2_k_dict', {})
            setup = st.session_state.get('s2_setup', {})
            _is_clhs = setup.get('is_clhs', False)
            
            # --- THỰC THI RẢI ĐIỂM THỰC TẾ TRƯỚC KHI LƯU ---
            if st.session_state.get('s2_final_n_est') is None:
                with st.spinner(f"Đang chạy thuật toán phân bổ tọa độ thực tế cho K = {current_target}... (Quá trình này có thể mất 10-30s)"):
                    core_target = current_target
                        
                    X_pca = st.session_state['s2_X_pca']
                    X_cat = st.session_state.get('s2_X_cat')
                    coords_v = st.session_state['s2_coords_valid']
                    meta = st.session_state['s2_meta']
                    
                    # Truyền cờ flag boolean thẳng vào hàm, bên trong hàm đã tự xử lý Dynamic D_max
                    is_dmax_enabled = setup.get('d_max_val')
                    
                    boundary_path = os.path.join(project_dir, 'sugarcane_boundary.geojson')
                    if _is_clhs:
                        dry_df = clhs_sampling(X_pca, coords_v, core_target, X_cat=X_cat, src_crs=meta.get('crs'), d_max=is_dmax_enabled, add_short_lags=setup.get('add_lags_val'), boundary_path=boundary_path)
                    else:
                        dry_df = k_means_sampling(X_pca, coords_v, core_target, src_crs=meta.get('crs'), d_max=is_dmax_enabled, add_short_lags=setup.get('add_lags_val'), boundary_path=boundary_path)
                    
                    st.session_state['s2_final_n_est'] = len(dry_df)
                    st.session_state['s2_dry_df'] = dry_df
            
            final_n = st.session_state['s2_final_n_est']
            st.info(f"💡 Hệ thống đã hoàn tất thuật toán rải điểm: Có **{final_n}** tọa độ đã sẵn sàng chốt hạ.")
            
            # Interactive Buttons (Action Bindings)
            st.markdown('**Dò tìm Vũng lầy (Live Analytics):**')
            c1, c2 = st.columns(2)
            
            next_economy = None
            if current_target in k_range:
                idx = k_range.index(current_target)
                if idx > 0:
                    import numpy as np
                    best_idx = int(np.argmin(scores[:idx]))
                    next_economy = k_range[best_idx]
            
            if next_economy is not None:
                std_idx = k_range.index(k_opt)
                eco_idx = k_range.index(next_economy)
                val_std = scores[std_idx]
                val_eco = scores[eco_idx]
                loss_pct = round(((val_eco - val_std) / val_std) * 100, 1) if val_std > 0 else 0
                
                if c1.button(f"⏪ Lùi K (Trích xuất: {next_economy} | ⚠️ Hao hụt {loss_pct}%)", width='stretch'):
                    st.session_state['s2_target_n'] = int(next_economy)
                    new_dict = k_dict.copy()
                    new_dict['economy'] = int(next_economy)
                    new_dict['economy_loss_pct'] = loss_pct
                    st.session_state['s2_k_dict'] = new_dict
                    
                    from modules.sampling_opt import plot_saturation_curve
                    new_fig = plot_saturation_curve(k_range, scores, new_dict, is_clhs=_is_clhs)
                    elbow_path = os.path.join(project_dir, 'stage2_elbow.png')
                    new_fig.savefig(elbow_path, dpi=120, bbox_inches='tight')
                    
                    # Force recompute DryRun
                    st.session_state['s2_final_n_est'] = None 
                    st.rerun()
            else:
                c1.button("⏪ Hết mức lùi", disabled=True, width='stretch')
                
            if c2.button(f"🔄 Lên đỉnh Tối ưu ({k_opt} mẫu)", width='stretch'):
                st.session_state['s2_target_n'] = int(k_opt)
                new_dict = k_dict.copy()
                new_dict['economy'] = None 
                new_dict['economy_loss_pct'] = 0
                st.session_state['s2_k_dict'] = new_dict
                
                from modules.sampling_opt import plot_saturation_curve
                new_fig = plot_saturation_curve(k_range, scores, new_dict, is_clhs=_is_clhs)
                elbow_path = os.path.join(project_dir, 'stage2_elbow.png')
                new_fig.savefig(elbow_path, dpi=120, bbox_inches='tight')
                
                st.session_state['s2_final_n_est'] = None 
                st.rerun()

            st.markdown("---")
            if st.button(f"✅ GIAO DỊCH: Chốt mốc K={current_target} (Sinh {final_n} điểm thật) & Chuyển sang Bước C", type="primary", width='stretch'):
                sampled_df = st.session_state['s2_dry_df']
                os.makedirs(spl_dir, exist_ok=True)
                sampled_path = os.path.join(spl_dir, 'optimal_samples.csv')
                sampled_df.to_csv(sampled_path, index=False)
                
                _save_stage2_config(project_dir, {
                    'k_optimal': int(k_opt),
                    'min_mlr': int(st.session_state['s2_min_mlr']),
                    'min_rf': int(st.session_state['s2_min_rf']),
                    'n_pcs': int(st.session_state['s2_n_pcs']),
                    'area_ha': round(st.session_state['s2_area_ha'], 1),
                    'model_type': setup.get('model_type'),
                    'pca_variance': float(explained),
                    'original_features': int(pca_pack.get('original_features', len(raster_files))),
                    'ui_state': {
                        'algo_choice': st.session_state.get('s2_algo_choice'),
                        'k_min': st.session_state.get('s2_k_min_val'),
                        'k_max': st.session_state.get('s2_k_max_val'),
                        'k_step': st.session_state.get('s2_k_step_val'),
                        'd_max': st.session_state.get('s2_d_max_val'),
                        'add_lags': st.session_state.get('s2_add_lags_val')
                    }
                })
                
                sampling_config = {
                    'model_type': setup.get('model_type'),
                    'n_samples': int(current_target),
                    'k_optimal_elbow': int(k_opt),
                    'n_pcs': int(st.session_state['s2_n_pcs'])
                }
                with open(os.path.join(project_dir, 'sampling_config.json'), 'w') as f:
                    json.dump(sampling_config, f, indent=4)
                    
                st.session_state['s2_coord_done'] = True
                st.rerun()

    # =====================================================
    # C. HIỂN THỊ KẾT QUẢ TỌA ĐỘ VÀ MAP
    # =====================================================
    if st.session_state.get('s2_coord_done'):
        st.markdown("---")
        st.subheader("C. Bảng vàng Xác nhận & Dữ liệu Tọa độ")
        
        sampled_path = os.path.join(spl_dir, 'optimal_samples.csv')
        if os.path.exists(sampled_path):
            import pandas as pd
            sampled_df = pd.read_csv(sampled_path)
            
            st.success(f"""
            **MỚI XÁC NHẬN!**
            | Chỉ số | Giá trị |
            |---|---|
            | Lõi Bão hòa Yêu cầu | **{st.session_state.get('s2_target_n')}** mẫu lõi |
            | Tổng lượng Tọa độ Khống chế Không gian | **{len(sampled_df)}** điểm thực tế |
            """)
            
            st.markdown("### 📍 Danh sách Tọa độ Lấy mẫu")
            st.info("WGS84 (Lon/Lat) phù hợp nạp GPS trắc địa. X_UTM/Y_UTM phù hợp lưới nội suy.")
            st.dataframe(sampled_df, width='stretch')
            
            if st.button("📂 Mở Thư Mục Chứa Tọa Độ (Dành cho Đội Thực Địa)", type="primary", use_container_width=True):
                os.startfile(spl_dir)
    

    # Trạng thái PCA
    if os.path.exists(pca_model_path):
        st.markdown("---")
        st.success(f"✅ **PCA sẵn sàng cho Giai đoạn 3 & 4** — {len(pc_rasters)} PC rasters trong `pca/`")
