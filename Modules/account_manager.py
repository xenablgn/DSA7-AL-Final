from database import get_connection, hash_password  

def update_user_details(user_id, username=None, email=None, phone=None, password=None):  
    connection = get_connection()  
    if connection is None:  
        print("Database connection error.")  
        return  

    with connection.cursor() as cursor:  
        updates = []  
        if username:  
            updates.append("username = %s")  
        if email:  
            updates.append("email = %s")  
        if phone:  
            updates.append("phone = %s")  
        if password:  
            password_hash = hash_password(password)  
            updates.append("password_hash = %s")  

        updates_query = ", ".join(updates)  
        query = f"UPDATE users SET {updates_query} WHERE id = %s"  

        try:  
            values = [v for v in (username, email, phone, password_hash) if v is not None]  
            values.append(user_id)  
            cursor.execute(query, values)  
            connection.commit()  
        except Exception as e:  
            print(f"Error updating account details: {e}")  
            connection.rollback()  
    connection.close()