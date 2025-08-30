from sklearn.metrics import r2_score, root_mean_squared_error

def predict_eval(model, data_preprocessed, target_series, name) -> str:
    y_pred = model.predict(data_preprocessed)
    rmse = root_mean_squared_error(target_series, y_pred)
    r2 = r2_score(target_series, y_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")
    return rmse, r2, f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}"