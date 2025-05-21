# DECISION-TREE-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: VAMSI PUTTEPU

*INTERN ID*: CT04DN135

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTORE*: NEELA SANTOSH

## 🧠 Task 1: Decision Tree Classifier on Pokémon Dataset – Detailed Description

In the first task of the CODTECH Machine Learning Internship, we focused on implementing a Decision Tree Classifier using the `scikit-learn` library. The aim was to build a predictive model capable of classifying whether a Pokémon is **Legendary** based on its attributes. This task served as an introduction to supervised machine learning, specifically classification techniques, data preprocessing, model evaluation, and visualization.

We used the popular `pokemon.csv` dataset, which contains a variety of features for each Pokémon. These features include categorical data such as primary and secondary types (`Type 1`, `Type 2`) and numerical attributes like `HP`, `Attack`, `Defense`, `Speed`, and more. The dataset also includes a target binary column called `Legendary`, indicating whether a Pokémon is classified as Legendary (`True`) or not (`False`). This column served as our target label for training the Decision Tree model.

### 🧪 Data Preprocessing and Exploration

The project began with exploratory data analysis (EDA) using the `pandas` library. We first loaded the dataset and examined its structure, checking for missing values, data types, and the distribution of the target column. It was important to drop non-informative columns such as the Pokémon `Name` and `#` (index), which do not contribute to model prediction and can introduce unnecessary noise.

Next, we performed label encoding for categorical columns like `Type 1` and `Type 2`. Label encoding is a necessary preprocessing step for machine learning models, as scikit-learn does not support string-type features. Using the `LabelEncoder` class from scikit-learn, we transformed these string labels into numeric representations.

### 🧠 Model Training

We defined our feature matrix `X` by excluding the target column `Legendary`, which was assigned to the label vector `y`. We then split the dataset into training and testing subsets using the `train_test_split()` function, ensuring 80% of the data was used for training and 20% for evaluation. This helped us measure the generalization performance of our model.

For model training, we used the `DecisionTreeClassifier` from scikit-learn with a controlled `max_depth` to prevent overfitting. Decision trees are easy-to-understand models that work well with both categorical and numerical data, making them a good choice for this task. After fitting the model on the training set, we used it to predict the labels for the test data.

### 📈 Model Evaluation

We evaluated the model using several classification metrics. The accuracy score provided a basic understanding of how well the model performed. However, we went further by calculating the **confusion matrix**, **precision**, **recall**, and **F1-score** using the `classification_report()` function. These metrics offered a more detailed view of the model’s strengths and weaknesses, especially important for binary classification tasks like this one.

### 🌳 Model Visualization

One of the most powerful aspects of Decision Trees is their interpretability. We visualized the trained model using `plot_tree()` from the `sklearn.tree` module. This plot clearly showed the feature-based decisions the tree makes at each node. We could observe which attributes, such as `Speed`, `Attack`, or `Sp. Atk`, played a significant role in determining if a Pokémon is Legendary.

### 🗂 Project Output and Files

All the code, visualizations, and results were compiled in a well-documented Jupyter Notebook. Additional files included a `README.md` file that explains the project in detail and a `requirements.txt` file listing the necessary Python libraries. This ensures the project is reproducible and easy to run on any machine.

### ✅ Conclusion

This task provided hands-on experience with the entire lifecycle of a machine learning classification project, from loading data to model evaluation and visualization. It reinforced key concepts such as feature encoding, train/test split, model accuracy, and the importance of interpretability in machine learning. Completing this task successfully lays a strong foundation for more advanced projects in the internship, such as natural language processing, deep learning with CNNs, and recommendation systems.

