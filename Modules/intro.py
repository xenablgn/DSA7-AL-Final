import streamlit as st  
import streamlit.components.v1 as components  
from database import delete_user, create_user, authenticate  
import sys
#sys.path.append('E:/Epita/action_learning/DSA7-AL-Final-dev')


st.set_page_config(page_title="Enhanced App", page_icon=":rocket:")  

if 'logged_in' not in st.session_state:  
    st.session_state['logged_in'] = False  

if 'current_page' not in st.session_state:  
    st.session_state['current_page'] = "Home"  

def reset_page_state():  
    st.session_state.update({  
        'current_page': "Home",  
    })  

def navigate_to(page):  
    st.session_state.current_page = page  

def logout():  
    st.session_state['logged_in'] = False  
    st.session_state.pop('user', None)  
    reset_page_state()  
    st.experimental_rerun()  

def sidebar():  
    user = st.session_state.get('user')  

    st.sidebar.title("Navigation")  
    st.sidebar.button("Home", on_click=lambda: navigate_to("Home"))  

    if st.session_state.logged_in:  
        st.sidebar.success(f"Welcome, {user['username']}!")  

        st.sidebar.button("Manage Account", on_click=lambda: navigate_to("ManageAccount"))  

        if user['admin']:  
            st.sidebar.button("Admin Panel", on_click=lambda: navigate_to("AdminPanel"))  

        st.sidebar.button("Logout", on_click=logout)  

    if not st.session_state.logged_in:  
        st.sidebar.radio("Option", ["Login", "Sign Up"], key='auth_option')  

def home_page():  
    st.title("Welcome to the MVP App!")  
    st.write("This is the home page.")  

    if st.button("Let's Start"):  
        navigate_to("App")  # Navigate to the app page when the button is clicked  
        st.experimental_rerun()  # Force rerun to reload the app page  

def login_page():  
    st.title("Login to Your Account")  
    username_or_email = st.text_input("Username or Email")  
    password = st.text_input("Password", type="password")  

    if st.button("Log In"):  
        user = authenticate(username_or_email, password)  
        if user:  
            st.session_state['logged_in'] = True  
            st.session_state['user'] = user  
            st.success(f"Logged in successfully! Welcome, {user['username']}")  
            navigate_to("Home")  # Redirect to Home Page after successful login  
            st.experimental_rerun()  # Force rerun to reload the Home page  
        else:  
            st.error("Invalid username/email or password.")  

def sign_up_page():  
    st.title("Create a New Account")  
    username = st.text_input("Username")  
    email = st.text_input("Email")  
    phone = st.text_input("Phone")  
    password = st.text_input("Password", type="password")  
    confirm_password = st.text_input("Confirm Password", type="password")  

    if st.button("Sign Up"):  
        if password == confirm_password:  
            create_user(username, email, phone, password)  
            st.success("User created successfully!")  
        else:  
            st.error("Passwords do not match.")  

def load_page(page):  
    if page == "Home":  
        home_page()  
    elif page == "Login":  
        login_page()  
    elif page == "SignUp":  
        sign_up_page()  
    elif page == "ManageAccount":  
        from ManageAccount import manage_account_page  # Importing ManageAccount page  
        manage_account_page()  
    elif page == "AdminPanel":  
        from AdminPanel import admin_panel_page  # Importing AdminPanel page  
        admin_panel_page()  
    elif page == "App":  
        import app  # Importing the app page  
        app.app_page()  # Assuming app.py has a function called app_page()  

sidebar()  

if not st.session_state.logged_in:  
    auth_option = st.session_state.get('auth_option')  
    if auth_option == "Login":  
        load_page("Login")  
    elif auth_option == "Sign Up":  
        load_page("SignUp")  
else:  
    load_page(st.session_state.current_page)