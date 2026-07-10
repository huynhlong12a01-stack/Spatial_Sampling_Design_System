import streamlit as st
import os

# --- EAGER LOADING (HACK TỐC ĐỘ) ---
# Tải nạp trước toàn bộ các thư viện Máy học & Bản đồ nặng nề vào bộ nhớ đệm (RAM)
# Việc này giúp khi ấn mở các Giai đoạn sẽ mượt mà ngay tắp lự (Warm Start).
try:
    import geopandas as gpd
    import rasterio
    import sklearn
    import geemap
    import gstools as gs
    import joblib
except ImportError:
    pass

from modules.views.settings import render_settings
from modules.views.project_dashboard import render_dashboard

st.set_page_config(page_title="Hệ thống Lập bản đồ Đất & Nông nghiệp", page_icon="🌍", layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Khởi tạo trạng thái Routing
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'home'

base_dir = os.path.join(os.getcwd(), 'data', 'projects')
os.makedirs(base_dir, exist_ok=True)
existing_projects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# --- SIDEBAR: QUẢN LÝ DỰ ÁN ---
st.sidebar.title("🌍 Quản lý Không gian")

if st.sidebar.button("⚙️ Cài đặt Hệ thống"):
    st.session_state['current_view'] = 'settings'
    st.rerun()

if st.sidebar.button("🗑️ Quản lý & Thùng rác"):
    st.session_state['current_view'] = 'project_manager'
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Danh sách Dự án")
for proj in existing_projects:
    if st.sidebar.button(f"📁 {proj}"):
        st.session_state['active_project'] = proj
        st.session_state['project_dir'] = os.path.join(base_dir, proj)
        st.session_state['current_view'] = 'project_dashboard'
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Thêm Dự án mới")
new_proj_name = st.sidebar.text_input("Tên dự án (Không dấu, không khoảng trắng):", label_visibility="collapsed")
if st.sidebar.button("Khởi động"):
    if new_proj_name:
        if new_proj_name in existing_projects:
            st.sidebar.error("Tên dự án đã tồn tại!")
        else:
            new_dir = os.path.join(base_dir, new_proj_name)
            os.makedirs(os.path.join(new_dir, 'covariates'), exist_ok=True)
            os.makedirs(os.path.join(new_dir, 'samples'), exist_ok=True)
            os.makedirs(os.path.join(new_dir, 'outputs'), exist_ok=True)
            
            st.session_state['active_project'] = new_proj_name
            st.session_state['project_dir'] = new_dir
            st.session_state['current_view'] = 'project_dashboard'
            st.rerun()

# --- MAIN ROUTER ---

view = st.session_state['current_view']

if view == 'home':
    st.title("🌱 Ứng dụng Nội suy Regression Kriging (RK) cho Nông nghiệp")
    st.markdown("👈 **Vui lòng chọn Dự án ở thanh bên trái hoặc Tạo dự án mới để làm việc.**")

elif view == 'settings':
    render_settings()

elif view == 'project_dashboard':
    if 'active_project' in st.session_state:
        render_dashboard()
    else:
        st.session_state['current_view'] = 'home'
        st.rerun()

elif view == 'project_manager':
    from modules.views.project_manager import render_project_manager
    render_project_manager()

elif view == 'stage1a':
    from modules.views.stage1a_boundary import render_stage1a
    render_stage1a()
    if st.button("⬅️ Quay lại Dashboard"):
        st.session_state['current_view'] = 'project_dashboard'
        st.rerun()

elif view == 'stage1b':
    from modules.views.stage1b_polygonize import render_stage1b
    render_stage1b()
    if st.button("⬅️ Quay lại Dashboard"):
        st.session_state['current_view'] = 'project_dashboard'
        st.rerun()

elif view == 'stage1c':
    from modules.views.stage1c_covariates import render_stage1
    render_stage1()
    if st.button("⬅️ Quay lại Dashboard"):
        st.session_state['current_view'] = 'project_dashboard'
        st.rerun()

elif view == 'stage2':
    from modules.views.stage2_sampling import render_stage2
    render_stage2()
    if st.button("⬅️ Quay lại Dashboard"):
        st.session_state['current_view'] = 'project_dashboard'
        st.rerun()


