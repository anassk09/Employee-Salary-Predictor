import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Load and clean data
df = pd.read_csv(r'C:\Users\shari\OneDrive\Desktop\test\adult.csv')
df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Step 2: Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Step 3: Split into features and label
X = df.drop('income', axis=1)
y = df['income']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Define ANN Deep learning
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 7: Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 8: Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Step 9: Save model
model.save("salary_predictor_model.keras")

# Step 10: Save scaler for use in Streamlit
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
