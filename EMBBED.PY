# Install necessary packages if not already installed
# pip install snowflake-connector-python pandas sentence-transformers

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import snowflake.connector

# Step 1: Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 2: Snowflake connection setup
conn = snowflake.connector.connect(
    user='SIDHARTHABOOLEAN',
    password='Sidhartha@2003',
    account='KO48922',  # e.g., xy12345.ap-southeast-1
    warehouse='YCOMPUTE_WH',
    database='FRAUD',
    schema='FRAUD_SCHEMA'
)

cursor = conn.cursor()

# Step 3: Fetch claim descriptions
query = "SELECT CLAIM_ID, CLAIM_DESCRIPTION FROM INSURANCE_CLAIMS"
cursor.execute(query)
rows = cursor.fetchall()

# Convert to DataFrame
df = pd.DataFrame(rows, columns=['CLAIM_ID', 'CLAIM_DESCRIPTION'])

# Step 4: Generate embeddings
df['CLAIM_EMBEDDING'] = df['CLAIM_DESCRIPTION'].apply(lambda text: model.encode(text).tolist())

# Step 5: Update embeddings back into Snowflake
update_query = """
    UPDATE INSURANCE_CLAIMS
    SET CLAIM_EMBEDDING = %s
    WHERE CLAIM_ID = %s
"""

for _, row in df.iterrows():
    embedding_array = row['CLAIM_EMBEDDING']
    claim_id = row['CLAIM_ID']
    
    # Convert list to Snowflake ARRAY format
    embedding_str = "[" + ",".join(map(str, embedding_array)) + "]"
    
    # Execute update
    cursor.execute(update_query, (embedding_str, claim_id))

# Cleanup
conn.commit()
cursor.close()
conn.close()

print("✅ Embeddings successfully updated into Snowflake.")
