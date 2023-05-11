# TimeSeriesMedicalImageClassification

Step1: Request access to download the dataset here: https://drive.google.com/file/d/1rKGrW57Nr6AN-jOQOPOQzQiHk8LD3q2o/view?usp=sharing

Step2: Download the ms_cxr_t data and move "MS_CXR_T_temporal_image_classification_v1.0.0.csv" to the root of directory.

Step3: Run
python feature_extraction.py {feature_extractor}

for extracting features from any of the four extractors used.

Step4: Run
python logistic_regression_baseline.py {feature_extractor}

for running the logistic regression model using features from any of the densenet features.

Step5: Run
python model2.py {feature_extractor}

for running the model2 (five separate independent classifiers) using features from any of the densenet features.

Step6: Run
python model3.py {feature_extractor}

for running the model3 (combined model with loss masking) using features from any of the densenet features.
