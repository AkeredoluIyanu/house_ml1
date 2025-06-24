import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title('🏠 House Price Prediction')

# Load dataset or user-uploaded file
house_sale_file = st.file_uploader('Upload your own house sale data')

if house_sale_file is not None:
    # Train new model on uploaded data
    house_sale = pd.read_csv(house_sale_file).dropna()
    X = house_sale.drop(["SALES_PRICE"], axis=1)
    y = house_sale["SALES_PRICE"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)

    score = forest.score(X_test, y_test)
    st.write(f'We trained a Random Forest model on your data. Test set R² score: **{score:.2f}**')

else:
    # Load default data and pre-trained model
    house_sale = pd.read_csv('house sale1.csv').dropna()
    X = house_sale.drop(["SALES_PRICE"], axis=1)
    y = house_sale["SALES_PRICE"]

    with open('forest.pickle', 'rb') as forest_pickle:
        forest = pickle.load(forest_pickle)
    
    st.write('We used a pre-trained Random Forest model with a test score of **~0.90**.')

# Plot feature importance
st.subheader("🔍 Feature Importance")
fig, ax = plt.subplots()
sns.barplot(x=forest.feature_importances_, y=X.columns, ax=ax)
ax.set_title('Top Features Affecting House Prices')
plt.tight_layout()
st.pyplot(fig)

# User inputs for prediction
st.subheader("📊 Predict House Price")

area = st.selectbox('AREA', ['Chrompet', 'Karapakkam', 'KK Nagar', 'Velachery', 'Anna Nagar', 'Adyar', 'T Nagar'])
int_sqft = st.number_input('INT_SQFT', min_value=0)
n_bedroom = st.number_input('N_BEDROOM', min_value=0)
n_bathroom = st.number_input('N_BATHROOM', min_value=0)
n_room = st.number_input('N_ROOM', min_value=0)
sale_c = st.selectbox('SALE_COND', ['AdjLand', 'Partial', 'Normal Sale', 'AbNormal', 'Family'])
build_type = st.selectbox('BUILDTYPE', ['House', 'Commercial', 'Others'])
mzzone = st.selectbox('MZZONE', ['RL', 'RH', 'RM', 'C', 'A', 'I'])
reg_fee = st.number_input('REG_FEE', min_value=0)
commis = st.number_input('COMMIS', min_value=0)
park = st.selectbox('PARK_FACIL', ['Yes', 'No'])

# Map inputs to numerical values
area_map = {'Adyar':0, 'Anna Nagar':1, 'Chrompet':2, 'KK Nagar':3, 'Karapakkam':4, 'T Nagar':5, 'Velachery':6}
sale_map = {'AbNormal':0, 'AdjLand':1, 'Normal Sale':3, 'Family':2, 'Partial':4}
mzzone_map = {'A':0, 'C':1, 'I':2, 'RH':3, 'RL':4, 'RM':5}
build_map = {'Commercial':0, 'House':1, 'Others':2}
park_map = {'Yes':1, 'No':0}

# Create input vector
input_data = [[
    area_map[area],
    int_sqft,
    n_bedroom,
    n_bathroom,
    n_room,
    sale_map[sale_c],
    build_map[build_type],
    mzzone_map[mzzone],
    reg_fee,
    commis,
    park_map[park]
]]

# Predict
if st.button("Predict"):
    prediction = forest.predict(input_data)
    st.success(f"🏡 Estimated House Price: **${prediction[0]:,.2f}**")
