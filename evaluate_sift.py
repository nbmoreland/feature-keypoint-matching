# Nicholas Moreland
# 1001886051

from sklearn.svm import LinearSVC

# Train an SVM model using the SIFT features
def sift_svm(X_train, y_train):
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# Evaluate accuracy of the SVM model
def evaluate_sift(svm_model, X_test, y_test):
    accuracy = svm_model.score(X_test, y_test)
    print(f'SVM Accuracy: {accuracy:.2f}')
    return accuracy