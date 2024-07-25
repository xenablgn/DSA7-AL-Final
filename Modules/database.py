#

import mysql.connector
import bcrypt

MYSQL_USER = 'root'  # Change to your MySQL username
MYSQL_PASSWORD = 'password'  # Change to your MySQL password
MYSQL_HOST = 'localhost'
MYSQL_PORT = '3306'
MYSQL_DB = 'MVP'  # Change to your database name

def get_all_users():
    connection = get_connection()
    if connection is None:
        print("Database connection error.")
        return

    users = []
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, username, email, phone, admin FROM users")
        users = cursor.fetchall()
    connection.close()
    return users

def update_user_role(user_id, admin):
    connection = get_connection()
    if connection is None:
        print("Database connection error.")
        return

    with connection.cursor() as cursor:
        cursor.execute("UPDATE users SET admin = %s WHERE id = %s", (admin, user_id))
        connection.commit()
    connection.close()

def delete_user(user_id):
    connection = get_connection()
    if connection is None:
        print("Database connection error.")
        return

    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        connection.commit()
    connection.close()

def get_connection():
    try:
        connection = mysql.connector.connect(
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            database=MYSQL_DB,
        )
        return connection
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        return None

def create_users_table():
    connection = get_connection()
    if connection is None:
        print("Database connection error.")
        return

    with connection.cursor() as cursor:
        cursor.execute("""  
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                phone VARCHAR(15),
                password_hash VARCHAR(255) NOT NULL,
                admin BOOLEAN DEFAULT FALSE
            )
        """)
        connection.commit()
    connection.close()

def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def check_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_user(username, email, phone, password, admin=False):
    create_users_table()
    connection = get_connection()
    if connection is None:
        print("Database connection error.")
        return

    with connection.cursor() as cursor:
        try:
            password_hash = hash_password(password)
            cursor.execute("""  
                INSERT INTO users (username, email, phone, password_hash, admin)  
                VALUES (%s, %s, %s, %s, %s)
            """, (username, email, phone, password_hash, admin))
            connection.commit()
        except Exception as e:
            print(f"Error creating user: {e}")
            connection.rollback()
    connection.close()

def authenticate(username_or_email, password):
    connection = get_connection()
    if connection is None:
        print("Database connection error.")
        return None

    with connection.cursor() as cursor:
        cursor.execute("""  
            SELECT id, username, password_hash, admin FROM users WHERE (username=%s OR email=%s)
        """, (username_or_email, username_or_email))

        user = cursor.fetchone()
        if user is None:
            return None

        user_id, username, stored_password_hash, is_admin = user

        if check_password(password, stored_password_hash):
            return {"id": user_id, "username": username, "admin": is_admin}
        else:
            return None

    connection.close()

def update_user_details(user_id, username=None, email=None, phone=None, password=None):
    connection = get_connection()
    if connection is None:
        print("Database connection error.")
        return

    with connection.cursor() as cursor:
        updates = []
        values = []
        if username:
            updates.append("username = %s")
            values.append(username)
        if email:
            updates.append("email = %s")
            values.append(email)
        if phone:
            updates.append("phone = %s")
            values.append(phone)
        if password:
            password_hash = hash_password(password)
            updates.append("password_hash = %s")
            values.append(password_hash)
        values.append(user_id)

        updates_query = ", ".join(updates)
        query = f"UPDATE users SET {updates_query} WHERE id = %s"

        try:
            cursor.execute(query, tuple(values))
            connection.commit()
        except Exception as e:
            print(f"Error updating account details: {e}")
            connection.rollback()
    connection.close()
