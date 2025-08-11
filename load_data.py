import snowflake.connector

# 1. Snowflake connection
conn = snowflake.connector.connect(
    user='SIDHARTHA MAHARANA',
    password='sIDHARTHA@2003',
    account='BOOLEANDATASYS_PARTNER',   # e.g., xy12345.ap-southeast-1
    warehouse='COMPUTE_WH',
    database='FRAUD',
    schema='FARUD_SCHEMA'
)

cursor = conn.cursor()

# 2. Create table for unstructured text
cursor.execute("""
CREATE TABLE IF NOT EXISTS FRAUD (
    ID INT AUTOINCREMENT,
    FILE_NAME STRING,
    CONTENT STRING
)
""")

# 3. Read TXT file
file_path = r"C:\Users\sidhartha-BD\Desktop\Acclerattors\claim_fraud_detection\claims_unstructured.txt"  # Change to your actual file path
with open(file_path, "r", encoding="utf-8") as f:
    file_content = f.read()

# 4. Insert into Snowflake
cursor.execute("""
INSERT INTO UNSTRUCTURED_CLAIMS (FILE_NAME, CONTENT)
VALUES (%s, %s)
""", ("claims_unstructured.txt", file_content))

# 5. Commit and close
conn.commit()
cursor.close()
conn.close()

print("TXT file stored successfully in Snowflake!")
