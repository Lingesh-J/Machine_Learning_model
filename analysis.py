import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-2.5-flash-lite')

def generate_summary(result_df):
    prompt = f''' You are a data scientist expert.
    Here are the model results:
    
    {result_df.to_string()}
    
    Suggest:
    
    - Identify the best model
    - Explain why it is best?
    - sumarize the performance of the model'''
    
    response = model.generate_content(prompt)
    return response.text

def suggest_improvements(result_df):
    prompt = f'''You are a data scientist expert, Here are the model results:
    
    {result_df.to_string()}
    
    Suggest:
    -ways to improve the model performance
    -Hyperparameter tuning and give range of values in each parameter
    - better suitable algorithms for the data
    - Data processing improvements'''
    
    response = model.generate_content(prompt)
    
    return response.text