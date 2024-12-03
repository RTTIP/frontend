#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install pandas sqlalchemy psycopg2 matplotlib requests


# In[5]:


import pandas as pd

# Sample data for demonstration (replace with actual database queries)
sample_assets_data = {
    "asset_id": [1, 2, 3],
    "name": ["Server1", "Database1", "Workstation1"],
    "value": [50000, 70000, 30000],
    "criticality": [8, 9, 7]
}

sample_threats_data = {
    "threat_id": [101, 102],
    "asset_id": [1, 2],
    "threat_description": ["DDoS Attack", "Data Breach"],
    "risk": [8.5, 9.2]
}

# Convert to DataFrame
df_assets = pd.DataFrame(sample_assets_data)
df_threats = pd.DataFrame(sample_threats_data)

# Validate data consistency
print("Assets Data:\n", df_assets)
print("\nThreats Data:\n", df_threats)

# Cleaning data
df_assets['value'] = df_assets['value'].fillna(0)
df_assets['criticality'] = df_assets['criticality'].fillna(df_assets['criticality'].mean())
df_assets.drop_duplicates(inplace=True)
df_threats.drop_duplicates(inplace=True)

print("\nCleaned Assets Data:\n", df_assets)
print("\nCleaned Threats Data:\n", df_threats)


# In[6]:


# Normalize asset criticality
df_assets['normalized_criticality'] = (
    (df_assets['criticality'] - df_assets['criticality'].min()) /
    (df_assets['criticality'].max() - df_assets['criticality'].min())
)

# Merge assets and threats
df_combined = pd.merge(df_assets, df_threats, on='asset_id', how='left')

# Save preprocessed data for AI models
df_combined.to_csv('processed_data.csv', index=False)
print("\nPreprocessed Data:\n", df_combined)
print("\nPreprocessed data saved to 'processed_data.csv'")


# In[7]:


import requests

# Sample API for demonstration (replace with actual AI API URL)
AI_API_URL = "http://mock-api-url.com/api/predict-impact"  # Replace with actual API

# Prepare data for API request
data_payload = df_combined.to_dict(orient='records')

# Mock API request (replace with actual API integration in production)
try:
    response = {"predictions": [{"asset_id": 1, "impact": "High"}, {"asset_id": 2, "impact": "Critical"}]}
    print("\nMock AI Predictions:\n", response)
except Exception as e:
    print("Failed to connect to AI API. Mock response used:", e)


# In[8]:


# Format data for LLM input
df_combined['llm_summary'] = df_combined.apply(
    lambda row: f"Asset {row['name']} with criticality {row['criticality']} "
                f"and value {row['value']} is linked to threat {row['threat_description'] or 'None'}.",
    axis=1
)

# Save summaries for LLM input
df_combined[['asset_id', 'llm_summary']].to_csv('llm_input.csv', index=False)
print("\nFormatted Data for LLM:\n", df_combined[['asset_id', 'llm_summary']])
print("\nFormatted data saved to 'llm_input.csv'")


# In[9]:


import matplotlib.pyplot as plt

# Visualize asset risk distribution
df_combined['risk'] = df_combined['risk'].fillna(0)  # Ensure no missing values in risk
plt.hist(df_combined['risk'], bins=5, color='skyblue', edgecolor='black')
plt.title('Asset Risk Distribution')
plt.xlabel('Risk Level')
plt.ylabel('Number of Assets')
plt.grid(True)
plt.show()

# Generate a simple report
average_risk = df_combined['risk'].mean()
print(f"\nAverage Risk Level: {average_risk}")
print(f"Total Number of Assets: {len(df_assets)}")


# In[ ]:




