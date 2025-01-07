import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Đọc dữ liệu
data = pd.read_csv('diabetes.csv')

# Chọn các cột dữ liệu cần thiết (loại bỏ cột Outcome nếu không sử dụng cho phân cụm)
features = data.drop(columns=['Outcome'])

# 2. Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# 3. Tính Silhouette Score cho số cụm từ 2 đến 10
silhouette_scores = []
for i in range(2, 11):  # Số cụm tối thiểu là 2
    model = AgglomerativeClustering(n_clusters=i, linkage='ward')
    labels = model.fit_predict(scaled_data)
    
    # Tính toán Silhouette Score cho mỗi số cụm
    score = silhouette_score(scaled_data, labels)
    silhouette_scores.append(score)

# 4. Vẽ đồ thị Silhouette Score
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Số cụm')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

# In ra các điểm Silhouette Score để tham khảo
for i in range(2, 11):
    print(f"Số cụm {i}: Silhouette Score = {silhouette_scores[i - 2]:.4f}")

# 5. Chọn số cụm tối ưu (ví dụ chọn số cụm có Silhouette Score cao nhất)
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Số cụm tối ưu: {optimal_clusters}")

# 6. Thực hiện phân cụm với số cụm tối ưu
model = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
labels = model.fit_predict(scaled_data)

# 7. Thêm nhãn phân cụm vào dữ liệu
data['Cluster'] = labels

# 8. Tính toán thống kê cơ bản cho mỗi cụm
pd.set_option('display.max_columns', None)
cluster_stats = data.groupby('Cluster').mean()

# 9. In ra thống kê cơ bản
print("\nThống kê cơ bản cho các thuộc tính trong mỗi cụm:")
print(cluster_stats)

# 10. Trực quan hóa kết quả phân cụm trong không gian 2D (sử dụng PCA)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Vẽ scatter plot với các cụm được phân loại
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.title(f'Kết quả phân cụm (Số cụm = {optimal_clusters})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# Vẽ biểu đồ dendrogram với phương pháp Ward
linked = linkage(scaled_data, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=False)
plt.title("Dendrogram (Agglomerative Clustering with Ward's Method)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# Các thuộc tính cần vẽ boxplot
attributes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Vẽ boxplot cho mỗi thuộc tính trong mỗi cụm
for attr in attributes:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y=attr, data=data)
    plt.title(f'Boxplot of {attr} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(attr)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# 2. Chọn 2 thuộc tính: Glucose và BMI
data_subset = data[['Glucose', 'BMI']]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_subset)

# 3. Áp dụng phương pháp phân cụm
model = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')  # Giả sử bạn chọn số cụm là 3
labels = model.fit_predict(scaled_data)

# 4. Thêm nhãn phân cụm vào dữ liệu
data['Cluster'] = labels

# 5. Trực quan hóa kết quả phân cụm
plt.figure(figsize=(8, 6))
plt.scatter(data['Glucose'], data['BMI'], c=data['Cluster'], cmap='viridis', marker='o')
plt.title('Agglomerative Clustering (Glucose vs BMI)')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.colorbar(label='Cluster')
plt.show()

# 6. Tính toán và in ra trung bình của mỗi thuộc tính trong từng cụm
cluster_means = data.groupby('Cluster').mean()

print("\nTrung bình của các thuộc tính trong mỗi cụm:")
print(cluster_means)


# Vẽ boxplot cho mỗi thuộc tính trong mỗi cụm
for attr in attributes:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y=attr, data=data)
    plt.title(f'Boxplot of {attr} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(attr)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
