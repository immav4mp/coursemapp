import json
import mysql.connector
from collections import defaultdict

# Database connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",          # change if needed
    password="",          # your MySQL password
    database="coursemap_db"
)
cursor = conn.cursor(dictionary=True)

# Step 1: Fetch all questions
cursor.execute("SELECT * FROM questions")
rows = cursor.fetchall()

# Step 2: Build JSON structure like question_bank.json
data = defaultdict(lambda: defaultdict(list))

for row in rows:
    course_name = row["course_name"]
    test_type = row["test_type"]

    question = {
        "id": row["question_id"],
        "question": row["question_text"],
        "choices": json.loads(row["choices"]) if row["choices"] else [],
        "answer": row["answer"]
    }

    data[course_name][test_type].append(question)

# Step 3: Save to JSON file
with open("question_bank.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

cursor.close()
conn.close()

print("✅ Exported database questions back to question_bank.json")
