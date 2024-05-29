import streamlit as st
import numpy as np
import os
import pandas as pd
import pickle
import sklearn
import requests
import io
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Define the RMSE function again to use when loading the model
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Load necessary objects and model
@st.cache_data
def load_resources():
    base_path = './'  # Adjust this path if your files are not in the current directory
    resources = ['user_id_encoder.pkl', 'skin_type_encoder.pkl', 'skin_tone_encoder.pkl', 
                 'product_name_encoder.pkl', 'brand_name_encoder.pkl', 'standardscaler.pkl',
                 'GMF_NCF_model']
    
    # Check each resource for existence before loading
    for resource in resources:
        resource_path = f"{base_path}{resource}"
        if not os.path.exists(resource_path):
            raise FileNotFoundError(f"Expected resource not found: {resource_path}")

    # If all resources are confirmed to be present, proceed to load
    user_id_encoder = pickle.load(open('./user_id_encoder.pkl', 'rb'))
    skin_type_encoder = pickle.load(open(f"{base_path}skin_type_encoder.pkl", 'rb'))
    skin_tone_encoder = pickle.load(open(f"{base_path}skin_tone_encoder.pkl", 'rb'))
    product_name_encoder = pickle.load(open(f"{base_path}product_name_encoder.pkl", 'rb'))
    brand_name_encoder = pickle.load(open(f"{base_path}brand_name_encoder.pkl", 'rb'))
    scaler = pickle.load(open(f"{base_path}standardscaler.pkl", 'rb'))
    keras_model = load_model(f"{base_path}GMF_NCF_model", custom_objects={'rmse': rmse})

    return user_id_encoder, skin_type_encoder, skin_tone_encoder, product_name_encoder, brand_name_encoder, scaler, keras_model

# Usage
user_id_encoder, skin_type_encoder, skin_tone_encoder, product_name_encoder, brand_name_encoder, scaler, keras_model = load_resources()


# Streamlit user interface
st.title("Skincare Product Recommendation System")
st.markdown("An Undergraduate Project by: Fadhilat Elakamah Arobovi")
st.sidebar.header("User Input Features")

# User details input
author_id_index = st.sidebar.text_input("User ID Index", "10")
skin_tone = st.sidebar.selectbox("Skin Tone", options=['dark', 'deep', 'fair', 'fairLight', 'light', 'lightMedium', 'medium', 'mediumTan', 'olive', 'porcelain', 'rich', 'tan'])
skin_type = st.sidebar.selectbox("Skin Type", options=['combination', 'dry', 'normal', 'oily'])
budget = st.sidebar.number_input("Budget (USD)", min_value=0, value=50, step=5)

# Convert index input to actual user ID
try:
    author_id = user_id_encoder.classes_[int(author_id_index)]
except IndexError:
    author_id = None
    st.sidebar.error("Invalid User ID Index!")

# Load and prepare product data
base_path = './'
product_details = pd.read_csv(f"{base_path}unique_products.csv")
product_details['price_ws'] = product_details['price_usd_reviews']
product_details[['price_usd_reviews', 'rating_products', 'loves_count']] = scaler.transform(product_details[['price_usd_reviews', 'loves_count', 'rating_products']])

# On button click, predict the ratings
if st.sidebar.button("Recommend Products") and author_id is not None:
    user_details_specific = {
        'author_id': author_id,
        'skin_tone': skin_tone,
        'skin_type': skin_type
    }

    # Encode user inputs
    user_input = np.array([user_id_encoder.transform([author_id])])
    skin_tone_input = np.array([skin_tone_encoder.transform([skin_tone])])
    skin_type_input = np.array([skin_type_encoder.transform([skin_type])])

    default_review_embedding = np.zeros(768)  # Assuming embedding size of 768
    default_ingredients_embedding = np.zeros(768)

    # DataFrame to store predictions
    predictions_df = pd.DataFrame(columns=['Brand Name', 'Product Name', 'Price', 'Estimated Rating'])
    total_products = len(product_details)
    progress_bar = st.progress(0)
    count = 0

    # Predict ratings for each product
    for index, product in product_details.iterrows():
        brand_input = np.array([brand_name_encoder.transform([product['brand_name']])])
        product_name_input = np.array([product_name_encoder.transform([product['product_name']])])
        price_input = np.array([[product['price_usd_reviews']]])
        rating_input = np.array([[product['rating_products']]])
        loves_input = np.array([[product['loves_count']]])

        review_input = np.array([default_review_embedding])
        ingredients_input = np.array([default_ingredients_embedding])

        # Prepare inputs for the model
        model_inputs = {
            'user_input': user_input,
            'skin_tone_input': skin_tone_input,
            'skin_type_input': skin_type_input,
            'brand_input': brand_input,
            'product_name_input': product_name_input,
            'price_input': price_input,
            'rating_input': rating_input,
            'loves_input': loves_input,
            'review_input': review_input,
            'ingredients_input': ingredients_input
        }
        try:
            estimated_rating = keras_model.predict(model_inputs)[0][0]
        except Exception as e:
            st.error(f"Error in model prediction: {e}")
            raise


        new_row = pd.DataFrame({
            'Brand Name': [product['brand_name']],
            'Product Name': [product['product_name']],
            'Price': [product['price_ws']],  # Use the duplicate column for display
            'Estimated Rating': [estimated_rating]
        })

        # Concatenate the new row to the existing DataFrame
        predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
        count += 1
        progress_bar.progress(count / total_products)

    # Store results in session to filter later
    st.session_state['predictions_df'] = predictions_df
    st.write(predictions_df.sort_values(by='Estimated Rating', ascending=False).head(10))

# Filtering by budget
if st.sidebar.button("Filter by Budget"):
    if 'predictions_df' in st.session_state:
        filtered_df = st.session_state['predictions_df']
        filtered_df = filtered_df[filtered_df['Price'] <= budget]
        st.write(filtered_df.sort_values(by='Estimated Rating', ascending=False).head(10))
    else:
        st.error("Please generate predictions first.")
