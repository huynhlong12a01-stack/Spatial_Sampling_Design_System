import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import gstools as gs

def extract_values_to_points(df, raster_files, lon_col='Longitude', lat_col='Latitude'):
    """Trích xuất giá trị từ các layer GeoTIFF tại các tọa độ điểm khảo sát"""
    coords = list(zip(df[lon_col], df[lat_col]))
    features = {}
    
    for path in raster_files:
        name = path.split('/')[-1].split('\\')[-1].split('.')[0]
        with rasterio.open(path) as src:
            # src.sample trả về generator
            samples = [x[0] for x in src.sample(coords)]
            features[name] = samples
            
    df_extracted = pd.DataFrame(features)
    # Kết hợp với df gốc
    return pd.concat([df.reset_index(drop=True), df_extracted.reset_index(drop=True)], axis=1), list(features.keys())

def train_regression_logic(X, y, model_type, cv_folds=5, cat_indices=None, **kwargs):
    """Huấn luyện mô hình cơ sở để dự đoán Trend"""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    if 'Linear' in model_type:
        if cat_indices:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_indices)
                ],
                remainder='passthrough'
            )
            model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
        else:
            model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100), 
            max_depth=kwargs.get('max_depth', None),
            random_state=42
        )
        
    # Cross validation để đánh giá
    y_pred_cv = cross_val_predict(model, X, y, cv=cv_folds)
    rmse = root_mean_squared_error(y, y_pred_cv)
    r2 = r2_score(y, y_pred_cv)
    
    # Train toàn bộ để lấy Final Model
    model.fit(X, y)
    y_pred_full = model.predict(X)
    residuals = y - y_pred_full
    
    return model, rmse, r2, residuals, y_pred_cv

def plot_regression_scatter(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Thực tế (Actual)')
    ax.set_ylabel('Dự báo CV (Predicted)')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    return fig

def fit_and_plot_variogram(x_coords, y_coords, residuals, model_name='Spherical', fit_mode='auto', nugget=None, var=None, len_scale=None):
    """Tính toán Variogram thực nghiệm và khớp hàm lý thuyết"""
    # Tính thực nghiệm
    bin_edges, gamma = gs.vario_estimate((x_coords, y_coords), residuals)
    
    # Khớp (Fit) hàm lý thuyết
    models_dict = {
        'Spherical': gs.Spherical,
        'Exponential': gs.Exponential,
        'Gaussian': gs.Gaussian
    }
    
    TheoModel = models_dict.get(model_name, gs.Spherical)
    
    if fit_mode == 'auto':
        fit_model = TheoModel(dim=2)
        fit_model.fit_variogram(bin_edges, gamma, nugget=True)
        res_nugget, res_var, res_len_scale = fit_model.nugget, fit_model.var, fit_model.len_scale
    else:
        # Manual Mode
        fit_model = TheoModel(dim=2, var=var, len_scale=len_scale, nugget=nugget)
        res_nugget, res_var, res_len_scale = nugget, var, len_scale
        
    # Vẽ Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(bin_edges, gamma, color='black', label="Thực nghiệm (Empirical)")
    ax.plot(bin_edges, fit_model.variogram(bin_edges), color='blue', label=f"Lý thuyết ({model_name})")
    ax.set_xlabel('Khoảng cách (Lag distance)')
    ax.set_ylabel('Semivariance')
    ax.legend()
    ax.set_title(f'Semivariogram ({model_name}) | Nugget: {res_nugget:.3f}, Partial Sill: {res_var:.3f}, Range: {res_len_scale:.1f}')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return fit_model, fig, res_nugget, res_var, res_len_scale

def calculate_vif(X, feature_names):
    """Tính toán Variance Inflation Factor (VIF) để phát hiện đa cộng tuyến"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_values = []
    
    # Chuẩn hóa để tránh nhiễu do scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for i in range(X_scaled.shape[1]):
        y_i = X_scaled[:, i]
        X_other = np.delete(X_scaled, i, axis=1)
        # R^2 của biến hiện tại theo các biến còn lại
        model = LinearRegression().fit(X_other, y_i)
        r_sq = model.score(X_other, y_i)
        
        # Công thức VIF = 1 / (1 - R^2)
        vif = 1.0 / (1.0 - r_sq) if r_sq < 1 else np.inf
        vif_values.append(vif)
        
    vif_data["VIF"] = vif_values
    return vif_data
