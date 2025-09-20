import json
import mysql.connector

# --- Database connection ---
conn = mysql.connector.connect(
    host="localhost",
    user="root",          # change if needed
    password="",          # your MySQL password
    database="coursemap_db"
)
cursor = conn.cursor()

# --- Drop old table (optional) ---
cursor.execute("DROP TABLE IF EXISTS questions")

# --- Create new tablee ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS questions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question_id VARCHAR(50) NOT NULL UNIQUE,
    strand ENUM('STEM','ABM','HUMSS','TVL','GAS') NOT NULL,
    course_name VARCHAR(100),
    test_type ENUM('knowledge','aptitude','interest','personality','goal') NOT NULL,
    question_text TEXT NOT NULL,
    option_a TEXT,
    option_b TEXT,
    option_c TEXT,
    option_d TEXT,
    answer VARCHAR(10)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# --- Load JSON file ---
with open("question_bank.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Insert questions ---
sql = """
INSERT INTO questions (
    question_id, strand, course_name, test_type, question_text,
    option_a, option_b, option_c, option_d, answer
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    strand = VALUES(strand),
    course_name = VALUES(course_name),
    test_type = VALUES(test_type),
    question_text = VALUES(question_text),
    option_a = VALUES(option_a),
    option_b = VALUES(option_b),
    option_c = VALUES(option_c),
    option_d = VALUES(option_d),
    answer = VALUES(answer)
"""

rows_inserted = 0
for strand, tests in data.items():  # Loop strands (stem, abm, humss, tvl, gas)
    strand_upper = strand.upper()
    if strand_upper not in ["STEM","ABM","HUMSS","TVL","GAS"]:
        continue

    for test_type, questions in tests.items():
        for q in questions:
            qid = str(q.get("id"))
            qtext = q.get("question")
            course = q.get("course") or None
            option_a = q.get("option_a")
            option_b = q.get("option_b")
            option_c = q.get("option_c")
            option_d = q.get("option_d")
            answer = q.get("answer")

            cursor.execute(sql, (
                qid, strand_upper, course, test_type, qtext,
                option_a, option_b, option_c, option_d, answer
            ))
            rows_inserted += 1

conn.commit()
cursor.close()
conn.close()

print(f"✅ Imported/updated {rows_inserted} questions into coursemap_db.questions")
