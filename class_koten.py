from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score

class MNISTClassifier:
    def __init__(self):
        # MNISTデータセットをロード
        self.mnist = datasets.load_digits()
        self.X = self.mnist.images.reshape((len(self.mnist.images), -1))
        self.y = self.mnist.target

        # データセットをトレーニングセットとテストセットに分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

        # 特徴量の標準化
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_knn(self, n_neighbors=3):
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.knn.fit(self.X_train, self.y_train)

    def train_svm(self, kernel='linear'):
        self.svm = SVC(kernel=kernel)
        self.svm.fit(self.X_train, self.y_train)

    def train_random_forest(self, n_estimators=100):
        self.random_forest = RandomForestClassifier(n_estimators=n_estimators)
        self.random_forest.fit(self.X_train, self.y_train)

    def train_gradient_boosting(self, n_estimators=100):
        self.gradient_boosting = GradientBoostingClassifier(n_estimators=n_estimators)
        self.gradient_boosting.fit(self.X_train, self.y_train)

    def train_naive_bayes(self):
        self.naive_bayes = GaussianNB()
        self.naive_bayes.fit(self.X_train, self.y_train)

    def evaluate_r2(self):
        r2_scores = {
            "KNN": r2_score(self.y_test, self.knn.predict(self.X_test)),
            "SVM": r2_score(self.y_test, self.svm.predict(self.X_test)),
            "Random Forest": r2_score(self.y_test, self.random_forest.predict(self.X_test)),
            "Gradient Boosting": r2_score(self.y_test, self.gradient_boosting.predict(self.X_test)),
            "Naive Bayes": r2_score(self.y_test, self.naive_bayes.predict(self.X_test))
        }
        return r2_scores

# 使用例
classifier = MNISTClassifier()
classifier.train_knn()
classifier.train_svm()
classifier.train_random_forest()
classifier.train_gradient_boosting()
classifier.train_naive_bayes()

r2_scores = classifier.evaluate_r2()
print(r2_scores)
