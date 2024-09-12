# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\Swast\OneDrive\Desktop\DA Proj\Python_Diwali_Sales_Analysis\Diwali Sales Data.csv", encoding='unicode_escape')

# Step 2: Data Cleaning and Preparation
# Select relevant columns for clustering
df = df[['Age', 'Amount', 'Product_Category', 'Marital_Status']]

# Drop missing values if any
df.dropna(inplace=True)

# Convert categorical variables to numeric
# For 'Marital_Status', convert to 0 (Single) and 1 (Married)
df['Marital_Status'] = df['Marital_Status'].apply(lambda x: 1 if x == 'Married' else 0)

# Use LabelEncoder for 'Product_Category' as it has string values
le = LabelEncoder()
df['Product_Category'] = le.fit_transform(df['Product_Category'])

# Step 3: Standardize the Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert back to DataFrame for easier interpretation
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Step 4: Apply K-Means Clustering
# Using the elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of square

# Try different numbers of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Step 5: Fit K-Means with the Optimal Number of Clusters (let's assume 3 clusters based on the elbow curve)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 6: Analyze and Visualize the Clusters
# Visualize the clusters based on Age and Amount
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='Amount', hue='Cluster', data=df, palette='Set1')
plt.title('Customer Segments based on Age and Amount')
plt.grid(True)
plt.show()

# Optional: Profile each cluster to understand their characteristics
cluster_profile = df.groupby('Cluster').mean()
print(cluster_profile)
