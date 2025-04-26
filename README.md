import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Yale_Admission_Data_2010_2025.csv")

df = df.drop(columns=['Admission_Rate'])
df['AdmitRate (%)'] = (df['Admitted'] / df['Applicants']) * 100
df['Enrollment_Rate (%)'] = (df['Enrolled'] / df['Admitted']) * 100
df['Rejection_Rate (%)'] = ((df['Applicants'] - df['Admitted']) / df['Applicants']) * 100

df.to_csv("Yale_Admission_Data_Prepared.csv", index=False)

plt.plot(df['Year'], df['AdmitRate (%)'])
plt.title("Admission Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Admit Rate (%)")
plt.grid(True)
plt.show()

plt.scatter(df['Tuition_Fees'], df['AdmitRate (%)'])
plt.title("Tuition Fee vs Admission Rate")
plt.xlabel("Tuition Fee")
plt.ylabel("Admit Rate (%)")
plt.grid(True)
plt.show()

plt.scatter(df['Financial_Aid (%)'], df['AdmitRate (%)'])
plt.title("Financial Aid vs Admission Rate")
plt.xlabel("Financial Aid (%)")
plt.ylabel("Admit Rate (%)")
plt.grid(True)
plt.show()

correlation_matrix = df[['Tuition_Fees', 'Financial_Aid (%)', 'AdmitRate (%)']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

X = df[['Tuition_Fees', 'Financial_Aid (%)']]
y = df['AdmitRate (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("R² Score:", r2_score(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
