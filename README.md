# OIBSIP - Data Science Internship
Oasis Infobyte Data Science Internship Projects - June 2026 Cohort.

---

## Task 1: Iris Flower Classification
A machine learning project focused on classifying iris flowers into three distinct species (Setosa, Versicolor, and Virginica) based on their sepal and petal measurements.

### Implementation Details:
* **Model Used:** Logistic Regression model optimized for multi-class classification.
* **Workflow:** Ingested the standard Iris dataset, performed feature mapping, split data into training/testing subsets, and evaluated classification accuracy.
* **Outcome:** Successfully trained the model to predict flower types with clear accuracy metrics printed in the terminal.

---

## Task 2: Unemployment Analysis with Python
An Exploratory Data Analysis (EDA) project tracking the economic impacts on labor market metrics, with a focus on regional distribution and major timeline spikes (such as the 2020 economic lockdowns).

### Key Insights Uncovered:
* **Timeline Analysis:** Discovered a massive, sharp spike in national unemployment between April 2020 and May 2020, capturing the direct impact of pandemic lockdowns. Urban areas experienced a sharper, higher peak compared to rural sectors.
* **Regional Distribution:** Identified significant disparities between states; regions like Tripura and Haryana maintained much higher median unemployment rates, while states like Meghalaya and Odisha showed greater labor market stability.

---

### Technologies Used:
* **Language:** Python 3.14
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
---

## Task 4: Email Spam Detection
A machine learning and Natural Language Processing (NLP) project built to classify text communications into legitimate messages (`Ham`) or unsolicited advertisements (`Spam`).

### Implementation Details:
* **Preprocessing & Feature Extraction:** Utilized a `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) to strip English stop-words and convert unstructured raw text strings into structured, high-dimensional numerical feature vectors.
* **Model Used:** Trained a `MultinomialNB` (Multinomial Naive Bayes) classification network, which uses probabilistic tracking ideally suited for text word-count distributions.
* **Workflow:** Built a standalone standalone pipeline handling dataset mapping, feature extraction, evaluation splits, and real-time inference handling.

### Outcome:
Successfully verified the pipeline with custom test inputs, tracking high-precision tracking metrics and accurately separating regular context phrases from standard scam patterns in the console.
---

## Task 3: Car Price Prediction
A machine learning regression project designed to estimate continuous vehicle market valuations based on physical and manufacturing attributes.

### Implementation Details:
* **Preprocessing Pipeline:** Integrated a `ColumnTransformer` utilizing `OneHotEncoder` to process categorical strings (Brand, Fuel Type) while leaving continuous variables untouched.
* **Model Selection:** Deployed a `RandomForestRegressor` ensemble model to handle nonlinear feature configurations and minimize pricing variance.
* **Workflow:** Bundled data multi-transformation and model training stages into a unified Scikit-Learn `Pipeline` object to securely manage inference splits.

### Outcome:
Successfully verified continuous value predictions on custom out-of-sample feature data frames directly inside the runtime console environment.