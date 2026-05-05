# Hotel Booking Cancellation Classification

## Project Goal

The goal of this project was to build a machine learning model that can predict whether a hotel booking will be canceled or not. The target column for this project is `is_canceled`, which tells whether a reservation was canceled.

This is a **classification problem** because the model is predicting a category instead of a number. The two possible classes are:

- `0` = the booking was not canceled
- `1` = the booking was canceled

This is different from regression because regression predicts a continuous number, such as price, temperature, or revenue. Since this project is making a yes/no prediction, classification is the correct machine learning approach.

## Dataset Used

The dataset used for this project is the **Hotel Booking Demand** dataset from Kaggle.

The data includes information about hotel reservations, such as:

- Hotel type
- Lead time
- Arrival date
- Length of stay
- Number of adults, children, and babies
- Meal type
- Country
- Market segment
- Previous cancellations
- Booking changes
- Deposit type
- Customer type
- Average daily rate
- Special requests
- Whether the booking was canceled

This dataset works well for classification because it has a clear target column, `is_canceled`, and many useful features that can help explain cancellation behavior. For example, a person booking far in advance may behave differently from someone booking close to the arrival date. A customer with previous cancellations may also be more likely to cancel again.

## Train, Validation, and Test Split

The data was split into three parts:

- **Training set:** 70%
- **Validation set:** 15%
- **Test set:** 15%

The training set was used to teach the models patterns in the data. The validation set was used while building the project to compare models and decide which ones were performing best. The test set was saved until the end so the final models could be checked on data that had not already been used for model selection.

This split is important because a model can sometimes perform well on data it has already seen, but that does not always mean it will perform well on new data. Saving a separate test set gives a more honest idea of how the model might perform on future hotel bookings.

The original midterm project already created a 70% training split and a 30% testing split. For the final project, the remaining 30% was split in half to create the 15% validation set and 15% test set.

## Preprocessing the Data

The raw hotel booking dataset could not be used directly by every machine learning model. Some columns had missing values, some were text categories, and some numeric columns had very large outliers. Because of this, preprocessing was needed before training the models.

### Missing Numeric Values

Missing numeric values were filled with the median. The median was used instead of the mean because it is less affected by extreme values. This matters because hotel booking data can include unusual high values, such as very long lead times.

### Missing Categorical Values

Missing categorical values were filled with the word `"missing"`. This kept the rows in the dataset instead of deleting them. It also allowed the model to treat missing information as its own category.

### One-Hot Encoding

Categorical columns were converted using one-hot encoding. Machine learning models usually need numbers instead of text, so one-hot encoding turns categories into separate 0/1 columns.

### Log Transformation

Some count-like numeric columns were log-transformed using `log1p`. This was done for columns where most values are small but a few values are very large. The log transformation helps reduce the effect of extreme values so they do not overpower the model.

The `adr` column was not log-transformed because it can contain negative values, and log transformations do not work well with negative numbers.

### Scaling

Numeric columns were scaled so that large-number columns would not overpower smaller-number columns. This is especially important for Logistic Regression, K-Nearest Neighbors, and Support Vector Classifier because these models can be affected by the size and range of numeric values.

## Handling Class Imbalance

The target variable, `is_canceled`, was somewhat imbalanced. This means there were not perfectly equal numbers of canceled and not-canceled bookings. The imbalance was not extreme, but it was still important to consider.

Class weighting was used where the model supported it. Class weighting tells the model to pay more attention to the smaller class so it does not only focus on the majority class.

This was chosen instead of SMOTE because class weighting keeps the original data intact. SMOTE creates synthetic examples, which can be useful, but it also makes the project harder to explain. For this project, class weighting was a simpler and more conservative choice.

## Repository File List

This repository currently contains:

| File | Description |
|---|---|
| `README.md` | Main project overview and explanation. |
| `dataset_source.md` | Notes or link information for the Hotel Booking Demand dataset source. |
| `hotel_booking_final_notebook.ipynb` | Main Jupyter Notebook containing the preprocessing, model training, validation, testing, and model comparison work. |
| `ModelDiscussionTable.pdf` | PDF table discussing model performance and comparison. |
| `link_to_cleaned_dataset.md` | Link or notes for accessing the cleaned dataset used for the final project. |
| `ComparisonReport.pdf` | PDF report discussing model performance and comparison.  |

## Models Trained

Several classification models were trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors Classifier
- Support Vector Classifier
- Gaussian Naive Bayes
- Average Ensemble of the best three models

Each model was tested using the same validation and test sets so the comparison would be fair.

## How the Models Were Compared

The models were compared using classification metrics:

- **Accuracy:** How many total predictions were correct.
- **Precision:** When the model predicted a booking would be canceled, how often it was correct.
- **Recall:** Out of all the bookings that really were canceled, how many the model caught.
- **F1-score:** A combined score that balances precision and recall.
- **ROC-AUC:** How well the model separates canceled bookings from not-canceled bookings.

For all of these classification metrics, a higher score is better.

## Best Individual Model

The best individual model was the **Decision Tree Classifier** based on validation F1-score.

This makes sense because hotel cancellations often follow rule-like patterns. A booking may be more likely to be canceled if the lead time is long, if the customer has canceled before, if the deposit type is certain categories, if there are fewer special requests, or if the booking came through a certain market segment.

A decision tree works by splitting the data into decision rules. That fits this problem well because cancellation behavior can often be explained through combinations of conditions.

## Final Selected Model

The final selected model was the **Average Ensemble Best 3**.

This model combined the three strongest validation models by averaging their predicted cancellation probabilities. The reason for using an ensemble is that one model may not catch every pattern in the data. Different models can make different kinds of mistakes.

By averaging the top three models, the final prediction becomes more stable and less dependent on the weaknesses of one single model.

## Bayesian Comparison Model

Gaussian Naive Bayes was used as the Bayesian comparison model. This model predicts by calculating the probability that a booking belongs to each class.

Gaussian Naive Bayes was not the strongest model because it assumes features are mostly independent from each other. That assumption does not fit this dataset very well because hotel booking features are often related. For example, lead time may relate to market segment, deposit type may relate to cancellation behavior, and previous cancellations may relate to future cancellations.

The dataset also has many one-hot encoded columns after preprocessing, which does not fit Gaussian Naive Bayes as well as the tree-based and ensemble models.

## Final Conclusion

This project used machine learning to predict whether a hotel booking would be canceled. The data was cleaned, encoded, scaled, and split into training, validation, and test sets. Several classification models were trained and compared using accuracy, precision, recall, F1-score, and ROC-AUC.

The **Decision Tree Classifier** was the best individual model because it handled rule-like cancellation patterns well. The **Average Ensemble Best 3** was the best overall model because it combined the strongest models and created a more stable final prediction.

Final model summary:

- **Final selected model:** Average Ensemble Best 3
- **Strongest single model:** Decision Tree Classifier
- **Weakest comparison model:** Gaussian Naive Bayes

Overall, the ensemble model is the best choice for this project because it gives the strongest and most balanced performance for predicting hotel booking cancellations.
