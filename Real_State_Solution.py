import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
import pickle

st.title("Real Estate Price Prediction Models")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('final.csv')
    return df

df = load_data()

st.subheader("Data Preview")
st.dataframe(df.head())

# Prepare data
x = df.drop('price', axis=1)
y = df['price']

# Fix dtype for categorical columns (if necessary)
df['property_type_Condo'] = df['property_type_Condo'].astype(object)
df['property_type_Bunglow'] = df['property_type_Bunglow'].astype(object)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=x['property_type_Bunglow'], random_state=42
)

st.write(f"Training set shape: {x_train.shape}")
st.write(f"Test set shape: {x_test.shape}")

# -------- Linear Regression --------
st.header("Linear Regression Model")

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

train_pred_lr = lr_model.predict(x_train)
test_pred_lr = lr_model.predict(x_test)

train_mae_lr = mean_absolute_error(y_train, train_pred_lr)
test_mae_lr = mean_absolute_error(y_test, test_pred_lr)

st.write(f"Train MAE: ${train_mae_lr:,.2f}")
st.write(f"Test MAE: ${test_mae_lr:,.2f}")

# -------- Decision Tree --------
st.header("Decision Tree Model")

dt_model = DecisionTreeRegressor(max_depth=3, max_features=10, random_state=567)
dt_model.fit(x_train, y_train)

train_pred_dt = dt_model.predict(x_train)
test_pred_dt = dt_model.predict(x_test)

train_mae_dt = mean_absolute_error(y_train, train_pred_dt)
test_mae_dt = mean_absolute_error(y_test, test_pred_dt)

st.write(f"Train MAE: ${train_mae_dt:,.2f}")
st.write(f"Test MAE: ${test_mae_dt:,.2f}")

# Plot decision tree
fig = plt.figure(figsize=(20, 10))
tree.plot_tree(dt_model, feature_names=dt_model.feature_names_in_, filled=True)
plt.title("Decision Tree Structure")
plt.tight_layout()
fig.savefig("tree.png", dpi=300)
plt.close(fig)

st.image("tree.png", caption="Decision Tree Structure", use_container_width=True)

# -------- Random Forest --------
st.header("Random Forest Model")

rf_model = RandomForestRegressor(n_estimators=200, criterion='absolute_error', random_state=42)
rf_model.fit(x_train, y_train)

train_pred_rf = rf_model.predict(x_train)
test_pred_rf = rf_model.predict(x_test)

train_mae_rf = mean_absolute_error(y_train, train_pred_rf)
test_mae_rf = mean_absolute_error(y_test, test_pred_rf)

st.write(f"Train MAE: ${train_mae_rf:,.2f}")
st.write(f"Test MAE: ${test_mae_rf:,.2f}")

# -------- Use Pickle to Save and Load Decision Tree --------
st.header("Save and Load Decision Tree Model with Pickle")

pickle.dump(dt_model, open('RE_Model.pkl', 'wb'))
RE_Model = pickle.load(open('RE_Model.pkl', 'rb'))

# Pick a sample from train set for prediction
sample_index = 22
sample_features = np.array(x_train.loc[sample_index])
actual_price = y_train.iloc[sample_index]
predicted_price = RE_Model.predict([sample_features])[0]

st.write(f"Sample index: {sample_index}")
st.write(f"Actual Price: ${actual_price:,.2f}")
st.write(f"Predicted Price: ${predicted_price:,.2f}")

# Optionally show sample features
st.subheader("Sample Features")
st.write(x_train.loc[sample_index])

