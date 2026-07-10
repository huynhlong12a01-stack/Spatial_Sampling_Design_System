import streamlit as st
import os
import shutil

def render_project_manager():
    st.title("🗑️ Quản trị Danh mục & Thùng rác Dự án")
    st.write("Trung tâm quản lý, đổi tên, xóa nội dung và khôi phục các dự án Không gian của bạn.")
    
    base_dir = os.path.join(os.getcwd(), 'data', 'projects')
    trash_dir = os.path.join(os.getcwd(), 'data', 'trash')
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(trash_dir, exist_ok=True)
    
    existing_projects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    trashed_projects = [d for d in os.listdir(trash_dir) if os.path.isdir(os.path.join(trash_dir, d))]
    
    tab1, tab2 = st.tabs([f"📂 Dự án Đang chạy ({len(existing_projects)})", f"♻️ Thùng rác ({len(trashed_projects)})"])
    
    with tab1:
        st.subheader("Danh sách Dự án Active")
        if not existing_projects:
            st.info("Hiện không có Dự án nào trong hệ thống, hãy khởi tạo bên thanh Sidebar Trái!")
            
        for proj in existing_projects:
            with st.expander(f"📁 {proj}"):
                col_rn, col_del = st.columns(2)
                
                with col_rn:
                    new_name = st.text_input(f"Nhập Tên mới:", value=proj, key=f"rn_{proj}")
                    if st.button("✏️ Chốt Đổi Tên", key=f"btn_rn_{proj}"):
                        if new_name == proj:
                            st.warning("Tên không có sự thay đổi.")
                        elif new_name in existing_projects or new_name in trashed_projects:
                            st.error("Lỗi: Tên dự án này đã tồn tại (Có thể đang nằm trong Thùng Rác).")
                        else:
                            old_path = os.path.join(base_dir, proj)
                            new_path = os.path.join(base_dir, new_name)
                            try:
                                shutil.move(old_path, new_path)
                                # Nếu dự án này đang được Active, sửa lại Session luôn
                                if st.session_state.get('active_project') == proj:
                                    st.session_state['active_project'] = new_name
                                    st.session_state['project_dir'] = new_path
                                st.success(f"Đã đổi tên thành công: `{new_name}`. Vui lòng Tải lại trang (F5).")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Lỗi hệ điều hành: {e}")
                                
                with col_del:
                    st.write("🗑️ **Đưa vào Thùng Rác (Xóa Mềm)**")
                    verify_txt = st.text_input("Vui lòng gõ chữ `YES` bằng tiếng anh để mở khóa lệnh Xóa:", key=f"del_{proj}")
                    if st.button("Xóa Mềm (Soft Delete)", type="primary", key=f"btn_del_{proj}"):
                        if verify_txt.strip().upper() == "YES":
                            old_path = os.path.join(base_dir, proj)
                            trash_path = os.path.join(trash_dir, proj)
                            try:
                                shutil.move(old_path, trash_path)
                                if st.session_state.get('active_project') == proj:
                                    del st.session_state['active_project']
                                    del st.session_state['project_dir']
                                st.success(f"Đã cách ly `{proj}` vào Cõi mộng (Thùng rác). Vui lòng tải lại (F5).")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Lỗi di chuyển file: {e}")
                        else:
                            st.error("Bạn chưa gõ chính xác chữ YES in hoa.")
                            
    with tab2:
        st.subheader("Linh hồn Dự án bị bỏ rơi")
        if not trashed_projects:
            st.info("Thùng rác đang sạch sẽ.")
            
        for trash in trashed_projects:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.markdown(f"**🗑️ `{trash}`**")
            
            with col2:
                if st.button("♻️ Khôi Phục", key=f"rs_{trash}"):
                    trash_path = os.path.join(trash_dir, trash)
                    restore_path = os.path.join(base_dir, trash)
                    if os.path.exists(restore_path):
                        st.error(f"Tuyệt vọng: Đã có Dự án mới tên `{trash}` đang hoạt động. Không thể phục hồi đè lên.")
                    else:
                        shutil.move(trash_path, restore_path)
                        st.success(f"Đã cho `{trash}` đội mồ sống lại! (F5)")
                        st.rerun()
            with col3:
                verify_kill = st.text_input("Gõ `YES`", key=f"hdel_txt_{trash}", label_visibility="collapsed")
                if st.button("🔥 Xoá Vĩnh Viễn", type="primary", key=f"hdel_{trash}"):
                    if verify_kill.strip().upper() == "YES":
                        trash_path = os.path.join(trash_dir, trash)
                        import time
                        try:
                            shutil.rmtree(trash_path)
                            st.success(f"Dự án `{trash}` đã bốc hơi khỏi vũ trụ! (F5)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi phá huỷ: {e}")
                    else:
                        st.error("Chưa gõ YES.")
