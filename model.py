import numpy as np
import pandas as pd

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor)

from sklearn.metrics import (mean_squared_error, r2_score, 
                             accuracy_score, precision_score,
                             recall_score, f1_score)



key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model= genai.GenerativeModel('gemini-2.5-flash-lite')
from analysis import generate_summary, suggest_improvements

st.set_page_config('Machine Learning model', page_icon = '⚠️☠️🚨', layout='wide')

st.title('Machine Learning model using AI 🛠')
st.header('Streamlit app to get csv and target as input and perform ML algorithm ☢️]')

uploaded_file=st.file_uploader('Upload a CSV file here ❯❯❯❯ ', type=['csv'])

if uploaded_file:
    st.markdown('### Preview')
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    target = st.selectbox(':blue[Select target 🎯]', df.columns)
    
    st.write(f' :red[Target variable : ]{target}')
    
    if target:
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()
        
        
        # Preprocessing
        
        num_cols = X.select_dtypes(include='number').columns.to_list()
        cat_cols = X.select_dtypes(include='object').columns.to_list()
        
        
        # Missing value treatment
        
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        X[cat_cols] = X[cat_cols].fillna('Missing data')
        
        # Encoding 
        X = pd.get_dummies(data=X, columns=cat_cols, 
                           drop_first=True, dtype=int)
        
        # For categoric target
        if y.dtype == 'object':
            label = LabelEncoder()
            y = label.fit_transform(y)
            
        #Detect the problem type
        
        if df[target].dtype == 'object' or len(np.unique(y)) <= 20:
            problem_type = 'classification' 
            
        else:
            problem_type = 'Regression'
            
            st.write(f'### 🔍 Problem Type: {problem_type}')
            
            #Train Test Split
            
            xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=42, test_size=0.2)
            
            #Scaling
            #fit tranform on train data
            #transform in test data
            
            for i in xtrain.columns:
                s = StandardScaler()
                
                xtrain[i] = s.fit_transform(xtrain[[i]])
                xtest[i] = s.transform(xtest[[i]])
                
            # Models
            
            results = []
            
            if problem_type == 'Regression':
                models = {'Linear Regression' : LinearRegression(),
                          'Random Forest': RandomForestRegressor(random_state=42),
                          'Gradient Boosting': GradientBoostingRegressor(random_state=42)}
                
                for name, model in models.items():
                    model.fit(xtrain, ytrain)
                    ypred = model.predict(xtest)
                    
                    results.append({'Model name': name,
                                    'R2 Score': round(r2_score(ytest, ypred), 3),
                                    'MSE':round(mean_squared_error(ytest, ypred), 3),
                                    'RMSE': round(np.sqrt(mean_squared_error(ytest, ypred)), 3)})
            else:
                models={'Logistic Regression':LogisticRegression(),
                        'Random Forest cl':RandomForestClassifier(random_state=42),
                        'Gradient Boosting cl':GradientBoostingClassifier(random_state=42)}
                
                for name, model in models.items():
                    model.fit(xtrain, ytrain)
                    ypred=model.predict(xtest)
                    
                    results.append({'Model Name':name,
                                    'Accuracy score':round(accuracy_score(ytest,ypred),2),
                                    'Precision':round(precision_score(ytest,ypred,average='weighted'),2),
                                    'Recall':round(recall_score(ytest,ypred,average='weighted'),2),
                                    'f1-score':round(f1_score(ytest,ypred,average='weighted'),2)})
            result_df = pd.DataFrame(results)
            st.write('### :green[Results]')
            st.dataframe(result_df)
            
            if problem_type == 'Regression':
                st.bar_chart(result_df.set_index('Model name')['R2 Score'])
                st.bar_chart(result_df.set_index('Model name')['RMSE'])
            else:
                st.bar_chart(result_df.set_index('Model name')['Accuracy'])
                st.bar_chart(result_df.set_index('Model name')['F1-Score'])
                
                
            # AI Insights
            
            if st.button('Generate summary'):
                summary = generate_summary(result_df)
                st.write(':blue [AI-Generated summary]')
                st.write(summary)
                
            if st.button('Suggest improvements'):
                improvements = suggest_improvements(result_df)
                st.write(':blue [AI generated improvements]')
                st.write(improvements)
                        
                     
