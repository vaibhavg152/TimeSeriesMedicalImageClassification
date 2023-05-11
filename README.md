# TimeSeriesMedicalImageClassification

Download the dataset here: https://drive.google.com/file/d/1rKGrW57Nr6AN-jOQOPOQzQiHk8LD3q2o/view?usp=sharing

Run

**python feature_extraction.py {feature_extractor}**

for extracting features from any of the four extractors used.

Run

**python logistic_regression_baseline.py {feature_extractor}**

for running the logistic regression model using features from any of the densenet features.

Run

**python model2.py {feature_extractor}**

for running the model2 (five separate independent classifiers) using features from any of the densenet features.

Run

**python model3.py {feature_extractor}**

for running the model3 (combined model with loss masking) using features from any of the densenet features.
