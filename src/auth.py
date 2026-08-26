"""
用户认证模块
包含用户登录、权限验证等功能
"""

import streamlit as st
import yaml
import bcrypt


# 加载用户数据
def load_users():
    with open("./.streamlit/users.yaml", "r") as f:
        return yaml.safe_load(f)["users"]


def update_password(username, new_password, users):
    if username in users:
        hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
        users[username]["password"] = hashed.decode()
        with open("./.streamlit/users.yaml", "w") as f:
            yaml.safe_dump({"users": users}, f)
        return True
    return False


def hashpassword(password):
    # Hash a password for the first time
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed


# 验证用户
def authenticate(username, password, users):
    if username in users:
        stored_hash = users[username]["password"].encode()
        return bcrypt.checkpw(password.encode(), stored_hash)
    return False


def require_login():
    if not st.session_state.get("authenticated", False):
        login_render()


def require_role(role):
    require_login()
    if st.session_state.get("role") != role:
        st.error("无权限访问此页面")
        st.stop()


def login_render():
    users = load_users()

    st.title("🔐 登录")

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        if st.button("登录"):
            if authenticate(username, password, users):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = users[username]["role"]
                st.success(f"登录成功，欢迎 {users[username]['name']}！")
                st.rerun()
            else:
                st.error("用户名或密码错误")
                st.rerun()
        else:
            st.stop()
    else:
        st.sidebar.success(f"已登录为 {st.session_state.username}")
