import streamlit as st
import os
import glob

from modules.map_render import build_project_map
from streamlit_folium import st_folium

def render_dashboard():
    project_name = st.session_state.get('active_project', 'Chưa xác định')
    project_dir = st.session_state.get('project_dir', '')
    
    st.title(f"📁 Dashboard Dự án: `{project_name}`")
    st.write("Theo dõi tiến trình và quản trị các phân hệ của dự án theo chuẩn DSM.")
    
    # Render Mắt Thần (Map Preview)
    with st.expander("🌍 Khởi động Mắt Thần (Interactive Project Map)", expanded=False):
        st.info("💡 Mẹo: Bản đồ chứa nhiều lớp ảnh viền thám nặng. Nhấp nút bên dưới để bắt đầu Render.")
        if st.button("🚀 Kích hoạt Tải Bản đồ Không gian", type="primary", use_container_width=True):
            with st.spinner("Đang nén ảnh Vệ Tinh và ráp Vector... Xin chờ giây lát..."):
                try:
                    m = build_project_map(project_dir)
                    st_folium(m, width='stretch', height=550, returned_objects=[])
                except Exception as e:
                    st.error(f"Lỗi khởi tạo Mắt Thần: {e}")
                
    if not project_dir:
        st.error("Lỗi: Không tìm thấy thư mục dự án.")
        return
        
    # Check status
    cov_dir = os.path.join(project_dir, 'covariates')
    spl_dir = os.path.join(project_dir, 'samples')
    out_dir = os.path.join(project_dir, 'outputs')
    
    has_covariates = len(glob.glob(os.path.join(cov_dir, '*.tif'))) > 0
    has_sugarcane_raster = os.path.exists(os.path.join(project_dir, 'sugarcane_boundary.tif'))
    has_sugarcane_vector = os.path.exists(os.path.join(project_dir, 'sugarcane_boundary.geojson'))
    has_samples = os.path.exists(os.path.join(spl_dir, 'optimal_samples.csv'))

    
    st.markdown("---")
    
    # Khối 1a: Ranh giới Ruộng Mía
    col1, col2 = st.columns([1, 4])
    with col1:
        if has_sugarcane_raster or has_sugarcane_vector: st.success("✅ Hoàn thành")
        else: st.warning("⏳ Buộc phải hoàn thành")
    with col2:
        st.subheader("Giai đoạn 1a: Ranh giới Ruộng Mía")
        st.write("Upload ranh giới ruộng mía có sẵn hoặc sử dụng AI Vệ tinh để tự động phát hiện.")
        if st.button("Mở Giai đoạn 1a ➡️", key="btn_stage1a"):
            st.session_state['current_view'] = 'stage1a'
            st.rerun()
            
        with st.expander("📁 Xem tài sản (Assets)"):
            if has_sugarcane_raster or has_sugarcane_vector:
                if has_sugarcane_raster:
                    st.markdown("- 🗺️ `sugarcane_boundary.tif` (Bản đồ AI)")
                if has_sugarcane_vector:
                    if os.path.exists(os.path.join(project_dir, 'sugarcane_boundary.geojson')):
                        st.markdown("- 🌾 `sugarcane_boundary.geojson` (Ranh giới Mía)")
            else:
                st.caption("Chưa có")

    st.markdown("---")

    # Khối 1b: Vector hóa (Polygonize)
    col1, col2 = st.columns([1, 4])
    with col1:
        if has_sugarcane_vector: st.success("✅ Hoàn thành")
        elif not has_sugarcane_raster: st.error("❌ Khóa")
        else: st.warning("⏳ Chờ xử lý")
    with col2:
        st.subheader("Giai đoạn 1b: Vector hóa (Polygonize)")
        st.write("Chuyển đổi bản đồ Raster AI thành Vector (GeoJSON) để làm đường viền phân tích.")
        
        btn1b_disabled = not has_sugarcane_raster
        if st.button("Mở Giai đoạn 1b ➡️", key="btn_stage1b", disabled=btn1b_disabled):
            st.session_state['current_view'] = 'stage1b'
            st.rerun()

    st.markdown("---")

    # Khối 1c: Biến môi trường
    col1, col2 = st.columns([1, 4])
    with col1:
        if has_covariates: st.success("✅ Hoàn thành")
        elif not has_sugarcane_vector: st.error("❌ Khóa")
        else: st.warning("⏳ Chờ xử lý")
    with col2:
        st.subheader("Giai đoạn 1c: Tải Biến môi trường (Covariates)")
        st.write("Sử dụng Ranh giới Mía ở trên để kéo dữ liệu DEM, NDVI, Slope một cách siêu tiết kiệm.")
        
        btn1c_disabled = not has_sugarcane_vector
        if st.button("Mở Giai đoạn 1c ➡️", key="btn_stage1c", disabled=btn1c_disabled):
            st.session_state['current_view'] = 'stage1c'
            st.rerun()
            
        with st.expander("📁 Xem tài sản (Assets)"):
            cov_files = glob.glob(os.path.join(cov_dir, '*.*'))
            pca_dir = os.path.join(project_dir, 'pca')
            pca_files = glob.glob(os.path.join(pca_dir, '*.*')) if os.path.exists(pca_dir) else []
            
            all_assets = cov_files + pca_files
            if all_assets:
                for idx, ff in enumerate(all_assets):
                    icon = "🧬" if "pca" in ff.lower() else "🗺️"
                    st.markdown(f"- {icon} `{os.path.basename(ff)}`")
            else:
                st.caption("Trống (Chưa có tài sản nào)")
            
    st.markdown("---")
    
    # Khối 2
    col1, col2 = st.columns([1, 4])
    with col1:
        if has_samples: st.success("✅ Hoàn thành")
        else: st.warning("⏳ Chờ xử lý")
    with col2:
        st.subheader("Giai đoạn 2: Thiết kế Không gian Mẫu (Sampling Design)")
        st.write("Chạy thuật toán K-Means hạn chế ngân sách để tìm điểm tọa độ đi thực địa.")
        if st.button("Mở Giai đoạn 2 ➡️", key="btn_stage2"):
            st.session_state['current_view'] = 'stage2'
            st.rerun()
            
        with st.expander("📁 Xem tài sản (Assets)"):
            spl_files = glob.glob(os.path.join(spl_dir, '*.*'))
            pca_dir = os.path.join(project_dir, 'pca')
            pca_files = glob.glob(os.path.join(pca_dir, '*.*')) if os.path.exists(pca_dir) else []
            all_s2_files = spl_files + pca_files
            if all_s2_files:
                for idx, ff in enumerate(all_s2_files):
                    icon = "🧬" if "PC" in os.path.basename(ff) or "pca" in os.path.basename(ff) else "📍"
                    st.markdown(f"- {icon} `{os.path.basename(ff)}`")
            else:
                st.caption("Trống (Chưa có tài sản nào)")

    st.markdown("---")
    

