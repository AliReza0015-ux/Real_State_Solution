#  Real Estate Price Prediction

## Project Structure

- `final.csv` - The dataset used for training and evaluation.
- `model_training.ipynb` - Jupyter Notebook with full model training and evaluation.
- `RE_Model` - A serialized (`.pkl`) version of the trained Decision Tree model.
- `tree.png` - Decision tree visualization saved as an image.
- `README.md` - Project overview and instructions.

---

##  Models Used

### ✅ Linear Regression
- Baseline model
- Trained on feature-engineered dataset
- Evaluated using **Mean Absolute Error (MAE)**

### ✅ Decision Tree Regressor
- `max_depth=3`, `max_features=10`
- Visualized using `sklearn.tree.plot_tree`
- Exported with `pickle`

### ✅ Random Forest Regressor
- `n_estimators=200`
- Criterion: `absolute_error`
- Improved generalization over single tree

---

##  Libraries Required

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pickle`

You can install all dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn
Authur: Ali Reza Mohseni
