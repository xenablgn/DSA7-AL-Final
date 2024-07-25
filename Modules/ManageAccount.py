import streamlit as st  
from database import update_user_details  

def manage_account_page():  
    user = st.session_state['user']  
    st.title("Manage Account")  

    new_username = st.text_input("New Username", value=user['username'])  
    new_email = st.text_input("New Email")  
    new_phone = st.text_input("New Phone")  
    new_password = st.text_input("New Password", type="password")  
    confirm_password = st.text_input("Confirm New Password", type="password")  

    if st.button("Update Account Details"):  
        if new_password and new_password != confirm_password:  
            st.error("Passwords do not match.")  
        else:  
            update_user_details(user['id'], new_username, new_email, new_phone, new_password)  
            st.success("Account details updated successfully.")  

# Call the function when this file is run directly  
if __name__ == "__main__":  
    manage_account_page()