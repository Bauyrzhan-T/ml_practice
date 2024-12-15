import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import GEOparse

# Step 1: Load gene expression data from GEO
def load_geo_data(dataset_id):
    geo_data = GEOparse.get_GEO(geo=dataset_id)
    expression_data = geo_data.pivot_samples('VALUE')
    return expression_data

# GSE32863 and GSE75037 datasets contain 1:1 lung cancer and adjacent healthy-cell data
expression_data_32863 = load_geo_data("GSE32863")
expression_data_75037 = load_geo_data("GSE75037")

# Combine datasets
expression_data = pd.concat([expression_data_32863, expression_data_75037], axis=1)
samples = expression_data.columns

# Define labels; 1 for cancer, 0 for non-tumor samples.
# GSE32863: odd = non-tumor, even = lung cancer
# GSE75037: odd = lung cancer, even = non-tumor
labels_32863 = [0 if i % 2 == 1 else 1 for i in range(expression_data_32863.shape[1])]
labels_75037 = [1 if i % 2 == 1 else 0 for i in range(expression_data_75037.shape[1])]
labels = np.array(labels_32863 + labels_75037)

# Step 2: Perform normalization
scaler = StandardScaler()
data_normalized = scaler.fit_transform(expression_data.T)

# Step 3: Perform feature selection (top k genes by ANOVA F-statistic)
k = 100  # Adjust the number of top genes if needed
selector = SelectKBest(score_func=f_classif, k=k)
data_reduced = selector.fit_transform(data_normalized, labels)
selected_genes = selector.get_support(indices=True)

# Step 4: Perform classification using Random Forest
classifier = RandomForestClassifier(random_state=42)
scores = cross_val_score(classifier, data_reduced, labels, cv=5, scoring='accuracy')

# Train final model for evaluation
classifier.fit(data_reduced, labels)
predicted_labels = classifier.predict(data_reduced)

# Step 5: Evaluate performance
print("Cross-validated Accuracy:", scores.mean())
print("Classification Report:\n", classification_report(labels, predicted_labels))

# Step 6: Feature interpretability (genes retained)
print("Top Genes (by index):", selected_genes)

# Save reduced data and selected genes to CSV for further analysis (e.g. KEGG pathway analysis)
output_data = pd.DataFrame(data_reduced, columns=[expression_data.index[i] for i in selected_genes])
output_data['Label'] = labels
output_data.to_csv('reduced_gene_expression.csv', index=False)
print("Reduced gene expression data saved to 'reduced_gene_expression.csv'.")

# Step 7: Define a function for classifying new samples
def classify_new_sample(sample_id, scaler, selector, classifier):
    # Load the new sample from GEO
    sample_data = GEOparse.get_GEO(geo=sample_id)
    expression_values = sample_data.table.set_index('ID_REF')['VALUE']

    # Match genes in the new sample to the training data
    common_genes = expression_data.index.intersection(expression_values.index)
    new_sample = expression_values.loc[common_genes].values

    # Step 1: Normalize the new sample using the same scaler
    new_sample_normalized = scaler.transform([new_sample])

    # Step 2: Reduce dimensionality using the same feature selector
    new_sample_reduced = selector.transform(new_sample_normalized)

    # Step 3: Predict the category (0 = negative, 1 = positive)
    predicted_label = classifier.predict(new_sample_reduced)

    # Step 4: Output the result
    category = "Lung Cancer Positive" if predicted_label[0] == 1 else "Lung Cancer Negative"
    print(f"The new sample ({sample_id}) is classified as: {category}")

# Test algorithm by predicting lung-cancer for GSM1012828 data sample (should output cancer-positive)
classify_new_sample("GSM1012828", scaler, selector, classifier)