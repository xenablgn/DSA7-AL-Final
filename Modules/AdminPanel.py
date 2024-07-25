import streamlit as st  
from database import get_all_users, update_user_role, delete_user

def admin_panel_page():  
    st.title("Admin Panel")  
    password = st.text_input("Admin Password", type="password")  

    if password == "secret":  # Replace with a secure method in a real application  
        users = get_all_users()  

        st.write("### Users")  
        
        for user in users:  
            st.write(f"ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Phone: {user[3]}, Admin: {user[4]}")  
            if not user[4] and st.button(f"Make {user[1]} Admin"):  
                update_user_role(user[0], True)  
                st.success(f"{user[1]} has been made an admin!")  
                st.experimental_rerun()  # Refresh the page to reflect the changes  

        st.write("### Additional Admin Features")  

        # Feature: Display Number of Users  
        st.write(f"Total Users: {len(users)}")  

        # Feature: Search User by Username  
        search_username = st.text_input("Search User by Username")  

        if st.button("Search"):  
            searched_users = [user for user in users if search_username.lower() in user[1].lower()]  
            if searched_users:  
                for user in searched_users:  
                    st.write(f"ID: {user[0]}, Username: {user[1]}, Email: {user[2]}, Phone: {user[3]}, Admin: {user[4]}")  
            else:  
                st.warning("No users found with that username.")  

          
        delete_user_id = st.text_input("Delete User ID")  
        if st.button("Delete User"):  
            if any(user[0] == int(delete_user_id) for user in users):  
                delete_user(int(delete_user_id))  
                st.success(f"User with ID {delete_user_id} has been deleted.")  
                st.experimental_rerun()  # Refresh the page to reflect the changes  
            else:  
                st.error("User ID not found.")  
        
        # Feature: Change User Role  
        target_user_id = st.text_input("Change Role - User ID")  
        new_admin_status = st.radio("New Role", ("User", "Admin"))  
        if st.button("Change Role"):  
            is_admin = True if new_admin_status == "Admin" else False  
            if any(user[0] == int(target_user_id) for user in users):  
                update_user_role(int(target_user_id), is_admin)  
                st.success(f"User with ID {target_user_id} role has been changed to {'Admin' if is_admin else 'User'}.")  
                st.experimental_rerun()  # Refresh the page to reflect the changes  
            else:  
                st.error("User ID not found.")  
        
    else:  
        st.error("Incorrect admin password!")

if __name__ == "__main__":  
    admin_panel_page()     