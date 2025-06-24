import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open('forest.pickle', 'rb') as f:
    forest = pickle.load(f)

# Feature names used during training
feature_names = ['AREA', 'INT_SQFT', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM',
                 'SALE_COND', 'BUILDTYPE', 'MZZONE', 'REG_FEE', 'COMMIS', 'PARK_FACIL']

fig, ax = plt.subplots()
sns.barplot(x=forest.feature_importances_, y=feature_names, ax=ax)
plt.title('Feature Importance for House Price Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')


# App header
st.title('House Price Prediction')
st.write("This app uses inputs to predict the price of a house.")

# User Inputs
area = st.selectbox('AREA', ['Chrompet','Karapakkam','KK Nagar','Velachery','Anna Nagar','Adyar','T Nagar'])
int_sqft = st.number_input('INT_SQFT', min_value=0)
n_bedroom = st.number_input('N_BEDROOM', min_value=0)
n_bathroom = st.number_input('N_BATHROOM', min_value=0)
n_room = st.number_input('N_ROOM', min_value=0)
sale_c = st.selectbox('SALE_COND', ['AdjLand', 'Partial','Normal Sale','AbNormal','Family'])
build_type = st.selectbox('BUILDTYPE', ['House', 'Commercial','Others'])
mzzone = st.selectbox('MZZONE', ['RL', 'RH','RM','C','A','I'])
reg_fee = st.number_input('REG_FEE', min_value=0)
commis = st.number_input('COMMIS', min_value=0)
park = st.selectbox('PARK_FACIL', ['Yes', 'No'])

# Encoding inputs
sale_c_map = {'AbNormal': 0, 'AdjLand': 1, 'Normal Sale': 3, 'Family': 2, 'Partial': 4}
mzzone_map = {'A': 0, 'C': 1, 'I': 2, 'RH': 3, 'RL': 4, 'RM': 5}
area_map = {'Adyar': 0, 'Anna Nagar': 1, 'Chrompet': 2, 'KK Nagar': 3, 'Karapakkam': 4, 'T Nagar': 5, 'Velachery': 6}
build_type_map = {'Commercial': 0, 'House': 1, 'Others': 2}
park_map = {'Yes': 1, 'No': 0}

# Apply mappings
sale_c = sale_c_map[sale_c]
mzzone = mzzone_map[mzzone]
area = area_map[area]
build_type = build_type_map[build_type]
park = park_map[park]

# Prediction
if st.button("Predict House Price"):
    new_data = [[area, int_sqft, n_bedroom, n_bathroom, n_room,
                 sale_c, build_type, mzzone, reg_fee, commis, park]]
    
    prediction = forest.predict(new_data)
    st.subheader("Predicted House Price:")
    st.write(f"${prediction[0]:,.2f}")
    st.image('feature_importance.png')
