df = pd.read_csv(f"../Data/data_for_NN_test2.txt")
X = df[[f"layer{i}" for i in range(1, IN_DIM + 1)]]
y = df[NAMES]
print(X.describe())
print(y.describe())

# X.describe().to_csv("X-summary.csv")
# y.describe().to_csv("y-summary.csv")

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1, random_state=42)

