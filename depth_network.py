import numpy as np
import time
# Function for z-score normalization
def normalize(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return (data - mean) / std_dev

learning_rate = 0.01
dp1_learning_rate = 1

# Dummy dataset
X_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
y_train = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=float)

# Normalize training data
X_train_normalized = normalize(X_train)
y_train_normalized = normalize(y_train)

start_time = time.time()
# Initialize model parameters
en_weights = np.random.rand(1)
en_biases = np.random.rand(1)
de_weights = np.random.rand(1)
de_biases = np.random.rand(1)

dp1we_en_weights = np.random.rand(1)
dp1we_en_biases = np.random.rand(1)
dp1bi_en_weights = np.random.rand(1)
dp1bi_en_biases = np.random.rand(1)

dp1we_de_weights = np.random.rand(1)
dp1we_de_biases = np.random.rand(1)
dp1bi_de_weights = np.random.rand(1)
dp1bi_de_biases = np.random.rand(1)
operations = 0

# Create a test dataset
X_test = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=float)
y_test = np.array([20, 22, 24, 26, 28, 30, 32, 34, 36, 38], dtype=float)

# Normalize test data
X_test_normalized = normalize(X_test)
y_test_normalized = normalize(y_test)

encoder_errors = []
# Training loop y_train to x_train
for epoch in range(100):
    # pass
    en_predictions = en_weights * X_train_normalized + en_biases
    de_predictions = de_weights * y_train_normalized + de_biases
    
    # Loss calculation (Mean Squared Error)
    en_loss = np.mean((en_predictions - y_train_normalized) ** 2)
    de_loss = np.mean((de_predictions - X_train_normalized) ** 2)

    # Backward pass (gradient calculation)
    en_dW = 2 * np.mean(X_train_normalized * (en_predictions - y_train_normalized))
    en_dB = 2 * np.mean(en_predictions - y_train_normalized)
    de_dW = 2 * np.mean(y_train_normalized * (de_predictions - X_train_normalized))
    de_dB = 2 * np.mean(de_predictions - X_train_normalized)

    # Predicted parameters
    en_weights_pred = en_weights - (learning_rate * en_dW)
    en_biases_pred  = en_biases - (learning_rate * en_dB)
    de_weights_pred = de_weights - (learning_rate * de_dW)
    de_biases_pred  = de_biases - (learning_rate * de_dB)

    if epoch > 1:
        for update_depth_epoch in range(5):
            du_en_weights = dp1we_en_weights * de_weights + dp1we_en_biases
            du_de_weights = dp1we_de_weights * en_weights + dp1we_de_biases
            du_en_biases  = dp1bi_en_weights * de_biases  + dp1bi_en_biases
            du_de_biases  = dp1bi_de_weights * en_biases  + dp1bi_de_biases

            #depth 1 pass loss calculation
            dp1_en_we_loss = np.mean((en_weights - en_weights_pred) ** 2)
            dp1_de_we_loss = np.mean((de_weights - de_weights_pred) ** 2)
            dp1_en_bi_loss = np.mean((en_biases - en_biases_pred) ** 2)
            dp1_de_bi_loss = np.mean((de_biases - de_biases_pred) ** 2)

            #depth 1 Backward pass (gradient calculation)
            dif_dp1we_en_weights = 2 * np.mean(de_weights * (en_weights - en_weights_pred))
            dif_dp1bi_en_weights = 2 * np.mean(de_biases * (en_biases - en_biases_pred))
            dif_dp1we_de_weights = 2 * np.mean(en_weights * (de_weights - de_weights_pred))
            dif_dp1bi_de_weights = 2 * np.mean(en_biases * (de_biases - de_biases_pred))
            dif_dp1we_en_biases = 2 * np.mean(en_weights - en_weights_pred)
            dif_dp1bi_en_biases = 2 * np.mean(en_biases - en_biases_pred)
            dif_dp1we_de_biases = 2 * np.mean(de_weights - de_weights_pred)
            dif_dp1bi_de_biases = 2 * np.mean(de_biases - de_biases_pred)
        
            # Update parameters
            dp1we_en_weights -= dp1_learning_rate * dif_dp1we_en_weights
            dp1bi_en_weights -= dp1_learning_rate * dif_dp1bi_en_weights
            dp1we_de_weights -= dp1_learning_rate * dif_dp1we_de_weights
            dp1bi_de_weights -= dp1_learning_rate * dif_dp1bi_de_weights
            dp1we_en_biases  -= dp1_learning_rate * dif_dp1we_en_biases
            dp1bi_en_biases  -= dp1_learning_rate * dif_dp1bi_en_biases
            dp1we_de_biases  -= dp1_learning_rate * dif_dp1we_de_biases
            dp1bi_de_biases  -= dp1_learning_rate * dif_dp1bi_de_biases
            operations+= 1


        #print(f"    Depth1 Encoder Weight Loss: {dp1_en_we_loss}")
        #print(f"    Depth1 Encoder Bias Loss: {dp1_en_bi_loss}")
        #print(f"    Depth1 Decoder Weight Loss: {dp1_de_we_loss}")
        #print(f"    Depth1 Decoder Bias Loss: {dp1_de_bi_loss}")
        # Apply updated depth 1 parameters as the new parameters for the main model
        en_weights, en_biases = du_en_weights, du_en_biases
        de_weights, de_biases = du_de_weights, du_de_biases
        
    else:
        # Apply predicted parameters for the main model
        en_weights, en_biases = en_weights_pred, en_biases_pred
        de_weights, de_biases = de_weights_pred, de_biases_pred

    # Use the model to make predictions on the test set
    en_predictions = en_weights * X_test_normalized + en_biases
    de_predictions = de_weights * y_test_normalized + de_biases

    # De-normalize predictions
    en_predictions_denormalized = en_predictions * np.std(y_test) + np.mean(y_test)
    de_predictions_denormalized = de_predictions * np.std(X_test) + np.mean(X_test)

    # Calculate MSE using NumPy
    en_mse_numpy = np.mean((y_test - en_predictions_denormalized) ** 2)
    de_mse_numpy = np.mean((X_test - de_predictions_denormalized) ** 2)

    # Calculate MSE using NumPy
    en_mse_numpy = np.mean((y_test - en_predictions_denormalized) ** 2)
    de_mse_numpy = np.mean((X_test - de_predictions_denormalized) ** 2)

    # Calculate RMSE using NumPy
    rmse_numpy = np.sqrt(en_mse_numpy)
    encoder_errors.append(rmse_numpy)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        print(f"    Encoder Loss: {en_loss}")
        print(f"    Decoder Loss: {de_loss}")
        print(f"Root Mean Squared Error (NumPy): {rmse_numpy}")
print(encoder_errors)
end_time = time.time()

# Display weights and biases
print(f"encoder Weights of the layer: {en_weights}, encoder Biases of the layer: {en_biases}")
print(f"decoder Weights of the layer: {de_weights}, decoder Biases of the layer: {de_biases}")

# Calculate the time difference
time_taken = end_time - start_time

print(f"Time taken: {time_taken} seconds")