import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

data = pd.read_csv('diabetes.csv')

features = data.drop(columns=['Outcome'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

silhouette_scores = []
for i in range(2, 11):  # Số cụm tối thiểu là 2
    model = AgglomerativeClustering(n_clusters=i, linkage='ward')
    labels = model.fit_predict(scaled_data)
    
    # Tính toán Silhouette Score cho mỗi số cụm
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Số cụm')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

for i in range(2, 11):
    print(f"Số cụm {i}: Silhouette Score = {silhouette_scores[i - 2]:.4f}")

optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Số cụm tối ưu: {optimal_clusters}")

model = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
labels = model.fit_predict(scaled_data)

data['Cluster'] = labels

pd.set_option('display.max_columns', None)
cluster_stats = data.groupby('Cluster').mean()

print("\nThống kê trung bình cho các thuộc tính trong mỗi cụm:")
print(cluster_stats)


pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.title(f'Kết quả phân cụm (Số cụm = {optimal_clusters})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


attributes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

rows = 2
cols = 4

fig, axes = plt.subplots(rows, cols, figsize=(15, 7))  # Kích thước toàn bộ cửa sổ

for i, attr in enumerate(attributes):
    row, col = divmod(i, cols)  # Tính vị trí hàng và cột
    sns.boxplot(x='Cluster', y=attr, data=data, ax=axes[row, col])  # Vẽ boxplot
    axes[row, col].set_title(f'{attr} by Cluster')  # Tiêu đề
    axes[row, col].set_xlabel('Cluster')  # Nhãn trục X
    axes[row, col].set_ylabel(attr)  # Nhãn trục Y
    axes[row, col].grid(True, linestyle='--', alpha=0.5)  # Lưới

plt.tight_layout()

plt.show()

data_subset = data[['Glucose', 'BMI']]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_subset)

model = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward') 
labels = model.fit_predict(scaled_data)

data['Cluster'] = labels

plt.figure(figsize=(8, 6))
plt.scatter(data['Glucose'], data['BMI'], c=data['Cluster'], cmap='viridis', marker='o')
plt.title('Agglomerative Clustering (Glucose vs BMI)')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.colorbar(label='Cluster')
plt.show()

cluster_means = data.groupby('Cluster').mean()

print("\nTrung bình của các thuộc tính trong mỗi cụm:")
print(cluster_means)


rows = 2
cols = 4

fig, axes = plt.subplots(rows, cols, figsize=(15, 7))  # Kích thước toàn bộ cửa sổ

for i, attr in enumerate(attributes):
    row, col = divmod(i, cols)  # Tính vị trí hàng và cột
    sns.boxplot(x='Cluster', y=attr, data=data, ax=axes[row, col])  # Vẽ boxplot
    axes[row, col].set_title(f'{attr} by Cluster')  # Tiêu đề
    axes[row, col].set_xlabel('Cluster')  # Nhãn trục X
    axes[row, col].set_ylabel(attr)  # Nhãn trục Y
    axes[row, col].grid(True, linestyle='--', alpha=0.5)  # Lưới

plt.tight_layout()

plt.show()
