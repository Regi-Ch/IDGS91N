
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

# Load data
df = pd.read_csv('Matriz.csv')

# Split features/labels
X = df[['glucosa', 'edad']]
y = df['etiqueta']

# Train/test split 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

for k in [3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[k] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

best_k = max(results, key=lambda k: results[k]['f1'])
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train_scaled, y_train)
best_preds = best_model.predict(X_test_scaled)
best_probs = best_model.predict_proba(X_test_scaled)[:,1]

cm = confusion_matrix(y_test, best_preds)

plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.colorbar()
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.savefig('matriz_confusion.png')
plt.close()

fpr, tpr, thresholds = roc_curve(y_test, best_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC - AUC = {:.3f}'.format(roc_auc))
plt.savefig('curva_roc.png')
plt.close()

print("Resultados por k:", results)
print("Mejor k:", best_k)
