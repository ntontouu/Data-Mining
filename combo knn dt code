# ΕΙΣΑΓΩΓΗ ΑΠΑΡΑΙΤΗΤΩΝ ΒΙΒΛΙΟΘΗΚΩΝ
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score, 
                             precision_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler

# ΦΟΡΤΩΣΗ ΤΟΥ ΣΥΝΟΛΟΥ ΔΕΔΟΜΕΝΩΝ
data = pd.read_csv('csv_result-Training Dataset.csv')
data.head()

# ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ: ΑΦΑΙΡΕΣΗ ΤΗΣ ΣΤΗΛΗΣ 'ID'
data = data.drop('id', axis=1)

# ΕΛΕΓΧΟΣ ΚΑΙ ΑΦΑΙΡΕΣΗ ΕΛΛΙΠΩΝ ΣΗΜΕΙΩΝ
data = data.dropna()  # Ή ΧΡΗΣΗ IMPUTATION ΑΝ ΧΡΕΙΑΖΕΤΑΙ

# ΔΙΑΧΩΡΙΣΜΟΣ ΤΩΝ ΔΕΔΟΜΕΝΩΝ ΣΕ ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ (X) ΚΑΙ ΣΤΟΧΟ (y)
X = data.drop('Result', axis=1)  # ΧΑΡΑΚΤΗΡΙΣΤΙΚΑ
y = data['Result']  # ΣΤΟΧΟΣ

# ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ ΤΩΝ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ
scaler = MinMaxScaler()  # ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ SCALER
X_normalized = scaler.fit_transform(X)  # ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ

# ΔΙΑΧΩΡΙΣΜΟΣ ΤΟΥ ΣΥΝΟΛΟΥ ΔΕΔΟΜΕΝΩΝ ΣΕ ΣΥΝΟΛΑ ΕΚΠΑΙΔΕΥΣΗΣ ΚΑΙ ΔΟΚΙΜΗΣ
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# ΑΡΧΙΚΟΠΟΙΗΣΗ ΚΑΙ ΕΚΠΑΙΔΕΥΣΗ ΤΟΥ ΤΑΞΙΝΟΜΗΤΗ ΔΕΝΔΡΟΥ ΑΠΟΦΑΣΕΩΝ
clf_dt = DecisionTreeClassifier(max_depth=30, random_state=1)

# CROSS VALIDATION ΓΙΑ DT
cv_scores_dt = cross_val_score(clf_dt, X_train, y_train, cv=10)
print(f'Αξιολόγηση cross validation Δέντρου Αποφάσεων: {cv_scores_dt}')
print(f'Μέση Ακρίβεια Δέντρου Αποφάσεων: {cv_scores_dt.mean():.3f}')

# Κατάρτιση του μοντέλου Δέντρου Αποφάσεων
clf_dt.fit(X_train, y_train)

# ΠΡΟΒΛΕΨΕΙΣ ΓΙΑ ΤΟ ΣΥΝΟΛΟ ΔΟΚΙΜΗΣ
y_pred_dt = clf_dt.predict(X_test)

# ΑΞΙΟΛΟΓΗΣΗ ΤΟΥ ΜΟΝΤΕΛΟΥ ΔΕΝΔΡΟΥ ΑΠΟΦΑΣΕΩΝ
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)

# ΕΚΤΥΠΩΣΗ ΤΩΝ ΜΕΤΡΙΚΩΝ ΑΞΙΟΛΟΓΗΣΗΣ ΓΙΑ ΤΟ ΔΕΝΤΡΟ ΑΠΟΦΑΣΕΩΝ
print(f'ΑΚΡΙΒΕΙΑ ΔΕΝΤΡΟΥ ΑΠΟΦΑΣΕΩΝ: {accuracy_dt:.3f}')
print(f'F1 SCORE ΔΕΝΤΡΟΥ ΑΠΟΦΑΣΕΩΝ: {f1_dt:.3f}')
print(f'ΑΝΑΚΛΗΣΗ ΔΕΝΤΡΟΥ ΑΠΟΦΑΣΕΩΝ: {recall_dt:.3f}')
print(f'ΑΚΡΙΒΗΣ ΔΕΝΤΡΟΥ ΑΠΟΦΑΣΕΩΝ: {precision_dt:.3f}')
print('ΑΝΑΦΟΡΑ ΤΑΞΙΝΟΜΗΣΗΣ ΔΕΝΤΡΟΥ ΑΠΟΦΑΣΕΩΝ:')
print(classification_report(y_test, y_pred_dt))

# Υπολογισμός πίνακα σύγχυσης για το Δέντρο Αποφάσεων
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Οπτικοποίηση του πίνακα σύγχυσης του Δέντρου Αποφάσεων
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=['Phishing', 'Non-Phishing'], yticklabels=['Phishing', 'Non-Phishing'])
plt.xlabel('Προβλεπόμενες Τιμές')
plt.ylabel('Πραγματικές Τιμές')
plt.title('Πίνακας Σύγχυσης - Δέντρο Αποφάσεων')
plt.show()

# ΑΡΧΙΚΟΠΟΙΗΣΗ ΚΑΙ ΕΚΠΑΙΔΕΥΣΗ ΤΟΥ ΤΑΞΙΝΟΜΗΤΗ k-NN
knn = KNeighborsClassifier(n_neighbors=10)  # ΕΠΙΛΟΓΗ ΤΗΣ ΤΙΜΗΣ Κ = 10

# CROSS VALIDATION ΓΙΑ knn
cv_scores_knn = cross_val_score(knn, X_train, y_train, cv=10)  # 10-fold cross-validation
print(f'Αξιολόγηση μέσω cross validation (k-NN): {cv_scores_knn}')
print(f'Μέση Ακρίβεια από cross validation (k-NN): {cv_scores_knn.mean():.3f}')

# Κατάρτιση του μοντέλου k-NN
knn.fit(X_train, y_train)

# ΠΡΟΒΛΕΨΕΙΣ ΓΙΑ ΤΟ ΣΥΝΟΛΟ ΔΟΚΙΜΗΣ
y_pred_knn = knn.predict(X_test)

# ΑΞΙΟΛΟΓΗΣΗ ΤΟΥ ΜΟΝΤΕΛΟΥ K-NN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)

# ΕΚΤΥΠΩΣΗ ΤΩΝ ΜΕΤΡΙΚΩΝ ΑΞΙΟΛΟΓΗΣΗΣ ΓΙΑ ΤΟ k-NN
print(f'ΑΚΡΙΒΕΙΑ K-NN: {accuracy_knn:.3f}')
print(f'F1 SCORE K-NN: {f1_knn:.3f}')
print(f'ΑΝΑΚΛΗΣΗ K-NN: {recall_knn:.3f}')
print(f'ΑΚΡΙΒΗΣ K-NN: {precision_knn:.3f}')
print('ΑΝΑΦΟΡΑ ΤΑΞΙΝΟΜΗΣΗΣ K-NN:')
print(classification_report(y_test, y_pred_knn))

# ΥΠΟΛΟΓΙΣΜΟΣ ΤΟΥ ΠΙΝΑΚΑ ΣΥΓΧΥΣΗΣ ΓΙΑ ΤΟ k-NN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# ΟΠΤΙΚΟΠΟΙΗΣΗ ΤΟΥ ΠΙΝΑΚΑ ΣΥΓΧΥΣΗΣ ΓΙΑ ΤΟ k-NN
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Phishing', 'Non-Phishing'], yticklabels=['Phishing', 'Non-Phishing'])
plt.xlabel('Προβλεπόμενες Τιμές')
plt.ylabel('Πραγματικές Τιμές')
plt.title('Πίνακας Σύγχυσης - k-NN')
plt.show()
