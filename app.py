import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
import argparse

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    train_data = df[df['Y'] < 38]
    test_data = df[df['Y'] >= 38]

    train_data = train_data.drop('Y', axis=1)
    test_data = test_data.drop('Y', axis=1)

    X_train = train_data.iloc[:, :df.shape[1]//2]  # Pixels 0 to 2047 (upper half)
    Y_train = train_data.iloc[:, df.shape[1]//2:]  # Pixels 2048 to 4095 (lower half)

    # Correctly split X_test and Y_test
    X_test = test_data.iloc[:, :df.shape[1]//2]  # Pixels 0 to 2047 (upper half)
    Y_test = test_data.iloc[:, df.shape[1]//2:]  # Pixels 2048 to 4095 (lower half)

    return X_train, Y_train, X_test, Y_test


def apply_pca(X_train, Y_train, X_test, Y_test, n_components_X=50, n_components_Y=50):
    pca_X = PCA(n_components=n_components_X)
    X_train_pca = pca_X.fit_transform(X_train)
    X_test_pca = pca_X.transform(X_test)

    pca_Y = PCA(n_components=n_components_Y)
    Y_train_pca = pca_Y.fit_transform(Y_train)
    Y_test_pca = pca_Y.transform(Y_test)
    
    return pca_X, pca_Y, X_train_pca, X_test_pca, Y_train_pca, Y_test_pca


def create_param_grid(): 
    # Change and experiment 
    param_grid = {
        "Random Forest": [200, 500],
        "Extra Trees": [200, 500],
    }
    return param_grid


def train_evaluate_models(X_train_pca, Y_train_pca, X_test_pca, Y_test_pca, param_grid, pca_Y):
    # Modify the param_grid and ESTIMATORS as needed for different models . Not done due to computational constraints
    # Or use different optimization techniques 
    '''
    param_grid = {
        "Extra Trees": {
            "n_estimators": [50, 100, 200],
            "min_samples_split": [2, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200, 500],
            "criterion": ['squared_error', 'absolute_error', 'friedman_mse']
        },
        "AdaBoost Regressor": {
            "n_estimators": [50, 100, 200,500],
            "learning_rate": [0.01, 0.1, 1.0]
        },
    }

    ESTIMATORS = {
        "Extra Trees": ExtraTreesRegressor(n_jobs=-1),
        "Random Forest": RandomForestRegressor(n_jobs=-1,bootstrap=True),
    }
    '''
    '''
    param_grid = {
        "estimator__n_estimators": [50, 100, 200, 500],
        "estimator__learning_rate": [0.01, 0.1, 1.0],
    }

    # Define the MultiOutputRegressor with AdaBoost as the base regressor
    ESTIMATORS = {
        "MultiOutput Regressor": MultiOutputRegressor(AdaBoostRegressor(random_state=0,n_estimators=500,learning_rate=0.1), n_jobs=-1),
    }

    # Train models, find optimal parameters, and predict
    results = {}
    for name, estimator in ESTIMATORS.items():
        print(f"\nPerforming GridSearchCV for {name}...")
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=3,  # 3-fold cross-validation
            scoring="neg_mean_squared_error",  # Use MSE for evaluation
            verbose=1,
            n_jobs=-1,
        )
        
        # Fit GridSearchCV
        grid_search.fit(X_train_pca, Y_train_pca)
        
        # Best model
        best_model = grid_search.best_estimator_
        print(f"Best Parameters for {name}: {grid_search.best_params_}")
        
        # Predict on test data
        Y_pred_pca = best_model.predict(X_test_pca)
        Y_pred = pca_Y.inverse_transform(Y_pred_pca)  # Inverse transform for visualization
        mse = mean_squared_error(Y_test, Y_pred)  # Calculate MSE
    '''
    # Define estimators with different models
    ESTIMATORS = {
        "Random Forest": [MultiOutputRegressor(RandomForestRegressor(n_estimators=n, random_state=0, n_jobs=-1)) for n in param_grid["Random Forest"]],
        "Extra Trees": [MultiOutputRegressor(ExtraTreesRegressor(n_estimators=n, random_state=0, n_jobs=-1)) for n in param_grid["Extra Trees"]],
    }

    # Train and evaluate models
    results = {}
    for name, models in ESTIMATORS.items():
        print(f"\nTraining {name} models...")
        for model in models:
            # Get the number of estimators for display
            n_estimators = model.estimator.n_estimators
            
            # Train the model
            print(f"Training {name} with n_estimators={n_estimators}...")
            model.fit(X_train_pca, Y_train_pca)
            
            # Predict on test data
            Y_pred_pca = model.predict(X_test_pca)
            Y_pred = pca_Y.inverse_transform(Y_pred_pca)  # Inverse transform for visualization
            
            # Compute MAE
            mae = mean_absolute_error(Y_test_pca, Y_pred_pca)  # Compute MAE on PCA-transformed data
            
            # Store results
            results[f"{name} (n_estimators={n_estimators})"] = {
                "model": model,
                "predictions": Y_pred,
                "mae": mae,
            }

    return results

# Function to plot results
def plot_results(results, X_test_pca, Y_test_pca, pca_X, pca_Y, image_shape=(64, 64)):
    n_faces = 5  # Number of faces to plot
    n_cols = 1 + len(results)  

    plt.figure(figsize=(2.0 * n_cols, 2.26 * n_faces))
    plt.suptitle("Face Completion with Multi-Output Estimators", size=16)

    for i in range(n_faces):
        # Inverse transform to reconstruct the full original face
        X_upper = pca_X.inverse_transform(X_test_pca[i])
        Y_lower_true = pca_Y.inverse_transform(Y_test_pca[i])
        true_face = np.hstack((X_upper, Y_lower_true))

        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
        if i == 0:
            sub.set_title("True Faces")
        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape), cmap="gray", interpolation="nearest")

        # Plot predictions from each model
        for j, (name, result) in enumerate(results.items()):
            Y_lower_pred = result["predictions"][i]
            completed_face = np.hstack((X_upper, Y_lower_pred))

            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
            if i == 0:
                sub.set_title(f"{name}\nMAE: {result['mae']:.2f}")
            sub.axis("off")
            sub.imshow(completed_face.reshape(image_shape), cmap="gray", interpolation="nearest")

    plt.show()

# Main function
def main():
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description="Run face completion using Random Forest and Extra Trees")
    parser.add_argument('--pca_components_X', type=int, default=50, help='Number of PCA components for X')
    parser.add_argument('--pca_components_Y', type=int, default=50, help='Number of PCA components for Y')
    parser.add_argument('--param_grid', type=str, default=None, help='Custom parameter grid for Random Forest and Extra Trees in the format: "Random Forest=200,500;Extra Trees=200,500"')

    args = parser.parse_args()

    # Load data
    X_train, Y_train, X_test, Y_test = load_data('data.csv')  # Ensure the correct path to data.csv

    pca_X, pca_Y, X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = apply_pca(
        X_train, Y_train, X_test, Y_test, n_components_X=args.pca_components_X, n_components_Y=args.pca_components_Y
    )

    param_grid = create_param_grid()

    if args.param_grid:
        try:
            custom_params = args.param_grid.split(';')
            for param in custom_params:
                model_name, values = param.split('=')
                values = list(map(int, values.split(',')))
                param_grid[model_name] = values
        except Exception as e:
            print(f"Error parsing custom parameter grid: {e}. Using default grid.")

    # Train models and get results
    results = train_evaluate_models(X_train_pca, Y_train_pca, X_test_pca, Y_test_pca, param_grid, pca_Y)

    # Plot results
    plot_results(results, X_test_pca, Y_test_pca, pca_X, pca_Y)

    # Display final MAE for each estimator
    print("\nFinal Mean Absolute Errors (MAE) values:")
    for name, result in results.items():
        print(f"{name}: MAE = {result['mae']:.2f}")

if __name__ == "__main__":
    main()