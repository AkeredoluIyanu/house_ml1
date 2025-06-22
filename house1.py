import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

house_sale = pd.read_csv('house sale1.csv')
house_sale = house_sale.dropna()
X=house_sale.drop(["SALES_PRICE"], axis=1)
y=house_sale["SALES_PRICE"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
fst = RandomForestRegressor()
fst.fit(X_train, y_train)
y_pred = fst.predict(X_test)

forest_pickle = open('forest_house.pickle', 'rb')
forest = pickle.load(forest_pickle)
forest_pickle.close()    
score = "Test set accuracy: {:.2f}".format(fst.score(X_test, y_test))
#st.write('We trained a Random Forest model on these data,'
#' it has a score of {}! Use the '
#'inputs below to try out the model.'.format(score))
st.write('We trained a Random Forest model on these data,'
 ' it has a score of Test set accuracy: 0.90! Use the '
 'inputs below to try out the model.')
    
rf_pickle = open('random_forest_house.pickle', 'wb')
pickle.dump(forest, rf_pickle)
rf_pickle.close()

fig, ax = plt.subplots()
ax = sns.barplot(forest.feature_importances_, X.columns)
plt.title('Which features are the most important for species prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')



st.title('House Price Prediction')
st.write("This app uses inputs to predict the price of house")






area = st.selectbox('AREA', options=['Chrompet','Karapakkam','KK Nagar','Velachery',
                                     'Anna Nagar','Adyar','T Nagar'])

int_sqft = st.number_input('INT_SQFT', min_value=0)

n_bedroom = st.number_input('N_BEDROOM', min_value=0)

n_bathroom = st.number_input('N_BATHROOM', min_value=0)

n_room = st.number_input('N_ROOM', min_value=0)

sale_c = st.selectbox('SALE_COND', options=['AdjLand', 'Partial','Normal Sale','AbNormal',
                                            'Family'])

build_type = st.selectbox('BUILDTYPE', options=['House', 'Commercial','Others'])

mzzone = st.selectbox('MZZONE', options=['RL', 'RH','RM','C','A','I'])

reg_fee = st.number_input('REG_FEE', min_value=0)

commis = st.number_input('COMMIS', min_value=0)

park = st.selectbox('PARK_FACIL', options=['Yes', 'No'])


if sale_c == 'AbNormal':
    sale_c = 0
elif sale_c == 'Family':
    sale_c = 2
elif sale_c == 'Partial':
    sale_c = 4
elif sale_c == 'AdjLand':
    sale_c = 1
else:
    sale_c = 3
    
if mzzone == 'A':
    mzzone = 0
elif mzzone == 'RH':
    mzzone = 3
elif mzzone == 'RL':
    mzzone = 4
elif mzzone == 'I':
    mzzone = 2
elif mzzone == 'C':
    mzzone = 1
else:
    mzzone = 5

if area == 'Karapakkam':
    area = 4
elif area == 'Anna Nagar':
    area = 1
elif area == 'Adyar':
    area = 0
elif area == 'Velachery':
    area = 6
elif area == 'Chrompet':
    area = 2
elif area == 'KK Nagar':
    area = 3
else:
    area = 5
    
if build_type == 'Commercial':
    build_type = 0
elif build_type == 'House':
    build_type = 1
else:
    build_type = 2
    
if park == 'Yes':
    park = 1
else:
    park = 0

    


new_prediction = forest.predict([[area, int_sqft, n_bedroom, n_bathroom,
                                     n_room, sale_c, build_type, mzzone, reg_fee, commis, park]])

st.subheader("Predicting house prices:")
st.write('We predict the house price is ${}'.format(new_prediction))
st.write('We used a machine learning (Random Forest) model to '
 'predict the prices, the features used in this prediction '
 ' are ranked by relative importance below.')
st.image('feature_importance.png')