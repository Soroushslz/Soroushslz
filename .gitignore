from gc import callbacks
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

# Define constants
BASE_DIR = 'Brain MRI Dataset of MS'
CLINICAL_DATA_FILE = 'patient info.xlsx'
TARGET_SHAPE = (64, 64, 64)  # Target shape for MRI images

# Load clinical data
clinical_data = pd.read_excel(CLINICAL_DATA_FILE)

# Drop the (PatientID) column
clinical_data = clinical_data.drop(columns=['PatientID'])

# Convert categorical variables to dummy/indicator variables
clinical_data = pd.get_dummies(clinical_data, drop_first=True)

# Extract features and target from dataset
X_clinical = clinical_data.iloc[:, :-1].values
Y = clinical_data.iloc[:, -1].values

# Preprocess the MRI dataset
preprocessed_nifti = np.load('preprocessed_mri_data.npy')
preprocessed_nifti = np.random.rand(len(Y), np.prod(TARGET_SHAPE) * 2)  # type: ignore # Placeholder

# Ensure the NIfTI data has a consistent shape
nifti_features_shape = (TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2], 2)

# Combine clinical and MRI dataset
X = np.hstack((X_clinical, preprocessed_nifti))

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Separate clinical and MRI data in train and test sets
num_clinical_features = X_clinical.shape[1]
X_train_clinical = X_train[:, :num_clinical_features]
X_train_mri = X_train[:, num_clinical_features:].reshape(-1, *nifti_features_shape)
X_test_clinical = X_test[:, :num_clinical_features]
X_test_mri = X_test[:, num_clinical_features:].reshape(-1, *nifti_features_shape)

# Ensure that X_train_mri and X_test_mri have consistent shapes
assert X_train_mri.shape[1:] == nifti_features_shape
assert X_test_mri.shape[1:] == nifti_features_shape

# Define F1-score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Convert to binary
        
        self.true_positives.assign_add(tf.reduce_sum(y_true * y_pred)) # type: ignore
        self.false_positives.assign_add(tf.reduce_sum((1 - y_true) * y_pred)) # type: ignore
        self.false_negatives.assign_add(tf.reduce_sum(y_true * (1 - y_pred))) # type: ignore

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1_score

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

# Define Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

# Create the CNN model
def create_mri_model(input_shape_mri):
    input_mri = tf.keras.layers.Input(shape=input_shape_mri)
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_mri)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)
    x = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)
    x = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling3D((2, 2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_mri, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                            metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

def create_clinical_model(input_shape_clinical):
    input_clinical = tf.keras.layers.Input(shape=input_shape_clinical)
    x = tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_clinical)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_clinical, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                            metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

input_shape_mri = TARGET_SHAPE + (2,)
input_shape_clinical = (X_clinical.shape[1],)

# Create models
mri_model = create_mri_model(input_shape_mri)
clinical_model = create_clinical_model(input_shape_clinical)

# Train the MRI model
mri_model.fit(X_train_mri, Y_train, epochs=50, batch_size=32, 
                validation_split=0.2, callbacks=[early_stopping], verbose=0)

# Train the Clinical model
clinical_model.fit(X_train_clinical, Y_train, epochs=50, batch_size=32, 
                validation_split=0.2, callbacks=[early_stopping], verbose=0)

# Get the logits from each model
mri_logits = mri_model.predict(X_train_mri)
clinical_logits = clinical_model.predict(X_train_clinical)

# Stack the logits together for a new model
X_train_fusion = np.hstack((mri_logits, clinical_logits))

# Define the fusion model
def create_fusion_model():
    input_fusion = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_fusion)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_fusion, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                            metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

# Create and train the fusion model
fusion_model = create_fusion_model()
fusion_model.fit(X_train_fusion, Y_train, epochs=50, batch_size=32, 
                validation_split=0.2, callbacks=[early_stopping], verbose=0)

def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def auc_roc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

# Predict using the trained individual models
mri_logits_train = mri_model.predict(X_train_mri)
mri_logits_test = mri_model.predict(X_test_mri)
clinical_logits_train = clinical_model.predict(X_train_clinical)
clinical_logits_test = clinical_model.predict(X_test_clinical)

# Stack the logits for fusion model evaluation
X_train_fusion = np.hstack((mri_logits_train, clinical_logits_train))
X_test_fusion = np.hstack((mri_logits_test, clinical_logits_test))

# Binarize the predictions
y_train_pred = (fusion_model.predict(X_train_fusion) > 0.5).astype(int)
y_test_pred = (fusion_model.predict(X_test_fusion) > 0.5).astype(int)

# Evaluate the fusion model
train_loss, train_accuracy, train_f1_score = fusion_model.evaluate(X_train_fusion, Y_train)
test_loss, test_accuracy, test_f1_score = fusion_model.evaluate(X_test_fusion, Y_test)
test_specificity = specificity(Y_test, y_test_pred)

# Print Accuracy and F1-score of the fusion model
print("Train Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
print("F1-score: {:.2f}".format(test_f1_score))
print("Specificity: {:.2f}".format(test_specificity))

# Create an explainer for the clinical model
explainer_clinical = shap.Explainer(clinical_model, X_train_clinical)
shap_values_clinical = explainer_clinical(X_test_clinical)

# Plot the SHAP values for the clinical model
shap.summary_plot(shap_values_clinical, X_test_clinical, feature_names=clinical_data.columns[:-1])

# flatten the MRI data before feeding it into SHAP
X_test_mri_flat = X_test_mri.reshape(X_test_mri.shape[0], -1)
explainer_mri = shap.Explainer(mri_model, X_test_mri_flat)
shap_values_mri = explainer_mri(X_test_mri_flat)

# visualizing the samples
shap.image_plot(shap_values_mri[:1], X_test_mri[:1])
