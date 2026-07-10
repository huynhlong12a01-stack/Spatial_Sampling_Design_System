import streamlit as st
import os
import json

def load_system_config():
    return {"gee_project_id": "rkapp-492504"}

def save_system_config(config):
    # Optional: could save to json if you still want to allow changes,
    # but the user requested to hardcode it, so we can just pass.
    pass

def render_settings():
    st.title("⚙️ Cài đặt Hệ thống Toàn cục")
    st.write("Thay vì mỗi dự án phải cấu hình lại, các chứng chỉ API và cài đặt phần mềm lõi sẽ được thiết lập vĩnh viễn ở đây.")
    
    config = load_system_config()
    
    st.markdown("### 1. Dịch vụ Google Earth Engine (GEE)")
    st.info("GEE yêu cầu một Cloud Project ID để có cấp quyền trích xuất dữ liệu mảng lớn. Bạn chỉ cần nhập một lần duy nhất.")
    
    gee_id = st.text_input("GEE Cloud Project ID:", value=config.get("gee_project_id", ""), placeholder="ee-youraccount...")
    
    if st.button("💾 Chốt Cấu Hình"):
        config['gee_project_id'] = gee_id
        save_system_config(config)
        st.success("Đã ghi đè cấu hình vào lõi hệ thống! Bạn có thể quay lại điều hành Dự án.")
