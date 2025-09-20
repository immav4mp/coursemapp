# database.py
import MySQLdb
from flask_mysqldb import MySQL

# Flask app will import this file
from flask import Flask

app = Flask(__name__)

# ✅ MySQL Config for phpMyAdmin
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'          # change if you have a different user
app.config['MYSQL_PASSWORD'] = ''          # put your MySQL password if set
app.config['MYSQL_DB'] = 'coursemap_db'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Initialize MySQL
mysql = MySQL(app)

# ✅ Function to test connection
def test_connection():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SHOW TABLES;")
        tables = cur.fetchall()
        print("✅ Connected! Tables in coursemap_db:", tables)
        cur.close()
    except Exception as e:
        print("❌ Connection failed:", e)


if __name__ == "__main__":
    test_connection()
