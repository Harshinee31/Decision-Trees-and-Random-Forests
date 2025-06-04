# Decision-Trees-and-Random-Forests
#Heart Disease Prediction using Decision Trees and Random Forests

This project demonstrates how to use **Decision Tree** and **Random Forest** models to classify whether a person is likely to have heart disease based on various medical attributes.

## Objective

- Train a Decision Tree Classifier and visualize it.
- Control overfitting by limiting tree depth.
- Train a Random Forest and compare performance.
- Interpret feature importances.
- Evaluate models using cross-validation.

## Dataset

The dataset used is the [Heart Disease UCI Dataset](https://www.kaggle.com/cherngs/heart-disease-cleveland-uci) which contains the following features:

| Feature        | Description                          |
|----------------|--------------------------------------|
| age            | Age in years                         |
| sex            | 1 = male; 0 = female                 |
| cp             | Chest pain type (4 values)           |
| trestbps       | Resting blood pressure               |
| chol           | Serum cholesterol in mg/dl           |
| fbs            | Fasting blood sugar > 120 mg/dl      |
| restecg        | Resting electrocardiographic results |
| thalach        | Maximum heart rate achieved          |
| exang          | Exercise-induced angina              |
| oldpeak        | ST depression induced by exercise    |
| slope          | Slope of the peak exercise ST segment|
| ca             | Number of major vessels (0-3) colored|
| thal           | Thalassemia type                     |
| target         | 1 = has heart disease, 0 = no disease|

> 📂 To use: Upload `heart.csv` to Google Colab.

##  Tools & Libraries

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

##  How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Upload the `heart.csv` file when prompted.
3. Run all cells to:
   - Load and explore the dataset
   - Train and evaluate models
   - Plot feature importances
   - Perform cross-validation

##  Model Performance

| Model                  | Accuracy (Test Set) | Cross-Validation Accuracy |
|------------------------|---------------------|----------------------------|
| Decision Tree (default)| ~0.73               | ~0.74                      |
| Decision Tree (pruned)| ~0.76               | ~0.77                      |
| Random Forest          | ~0.80+              | ~0.82                      |

> Note: Actual results may vary slightly depending on the random state and system environment.

##  Feature Importance (Random Forest)

The model shows which features contribute most to predicting heart disease, such as:
- `cp` (chest pain type)
- `thalach` (max heart rate)
- `oldpeak` (ST depression)

## Conclusion

Tree-based models provide interpretable and accurate predictions for medical classification problems. Pruning and ensemble methods like Random Forest improve performance and reduce overfitting.

---


