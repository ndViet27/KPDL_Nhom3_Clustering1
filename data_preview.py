import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


file_name = "C:\\Users\\THINKPAD\\Documents\\kpdl\\baitapcuoimon\\diabetes.csv"
df = pd.read_csv(file_name)
print(df.head(10))
print(" ")
df.info()

df['Outcome'] = df['Outcome'].astype('category',copy=False)
def convert_outcome(x):
    if x == 1:
        return 'Yes'
    else:
        return 'No'
df['Outcome'] = df['Outcome'].apply(convert_outcome)

df.info()
print("")
pd.set_option('display.max_columns', None)
print(df.describe())

#Biểu đồ histogram
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
plt.figure(figsize=(15, 10)) 
for i, column in enumerate(columns, 1):  
    plt.subplot(2, 4, i)  
    sns.histplot(df[column], bins=20, kde=True, color='blue')  
    plt.title(column)  
    plt.xlabel("")  
    plt.ylabel("")  

plt.tight_layout(h_pad=4.0)
plt.show()

#Biểu đồ nhiệt biểu hiện giá trị tương quan giữa các thuộc tính
d = df.drop(['Outcome'],axis=1)
correlation_matrix = d.corr()

plt.figure(figsize=(10, 8))  
sns.heatmap(
    correlation_matrix, 
    annot=True,         
    fmt=".2f",         
    cmap="Oranges", 
    linewidths=0.5
)
plt.title("Heatmap of Attribute Correlations", fontsize=16) 
plt.show()

#Biểu đồ số lượng của thuộc tính phân loại
dfg = df['Outcome'].value_counts().reset_index()
dfg.columns = ['Outcome', 'Quantity']

plt.figure(figsize=(8, 6))
plt.bar(dfg['Outcome'], dfg['Quantity'], color='blue', alpha=0.7)

plt.title('Quantity of Patients by Outcome', fontsize=14)
plt.xlabel('Outcome', fontsize=12)                        
plt.ylabel('Quantity', fontsize=12)                      

plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()


#Biểu đồ boxplot
plt.figure(figsize=(15, 8)) 

for i, col in enumerate(columns, 1):
    plt.subplot(2,4, i)
    sns.boxplot(x='Outcome', y=col, data=df)
    plt.title(f'{col} vs Outcome', fontsize=12) 
    plt.xlabel('Outcome')  
    plt.ylabel(col)

plt.tight_layout(h_pad=3.0)
plt.show()

# Tách dữ liệu theo Outcome
outcome_0 = df[df['Outcome'] == 'No']
outcome_1 = df[df['Outcome'] == 'Yes']

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.scatter(outcome_0['Glucose'], outcome_0['BMI'], color='green', label='Outcome No', alpha=0.6, edgecolors='k')
plt.scatter(outcome_1['Glucose'], outcome_1['BMI'], color='red', label='Outcome Yes', alpha=0.6, edgecolors='k')
plt.title("Relationship between Glucose and BMI by Outcome", fontsize=14)
plt.xlabel("Glucose (mg/dL)", fontsize=12)
plt.ylabel("BMI (kg/m²)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
