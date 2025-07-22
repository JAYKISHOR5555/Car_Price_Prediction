from car_price_model import *

# Load data
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

# Train models
tree_model = train_decision_tree(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Evaluate models
tree_metrics = evaluate_model(tree_model, X_test, y_test)
rf_metrics = evaluate_model(rf_model, X_test, y_test)

# Show results
print("Decision Tree R²:", tree_metrics["r2"])
print("Random Forest R²:", rf_metrics["r2"])

# Save best model
save_model(rf_model)
