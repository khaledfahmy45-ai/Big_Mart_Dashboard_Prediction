import pandas as pd 
import numpy as np 
import streamlit as st 
import joblib
import plotly.express as px

st.set_page_config(layout='wide')
st.markdown("<h1 style = 'text-align : center ; color : green;' > Big Mart Sales Prediction </h1>" , unsafe_allow_html=True)
st.write('''Welcome to the Big Mart Project.
Use the sidebar to navigate between:
- Data Analysis
- Prediction''')
df = pd.read_csv('model_big_mart.csv')
loaded_model = joblib.load('model_big_mart.pkl')
page = st.sidebar.selectbox('Select Page', ['Data Analysis', 'Prediction'])

if page == 'Data Analysis':
       st.title('Data Analysis')
       st.header('Dataset Describtion')
       st.markdown('''This dataset contains information about products sold across different Big Mart outlets.
                     The goal is to predict **Item Outlet Sales** based on product and store characteristics.''')
       df = pd.read_csv('D:/Data/Final_Project/model_big_mart.csv')
       st.subheader('Column Description')
       st.markdown('''
       - Item_Weight: Weight of the product.
       - Item_Fat_Content: Whether the product is low fat or not.
       - Item_Visibility: Percentage of total display area allocated to the product.
       - Item_Type: Category of the product either Dairy or Fruits etc.
       - Item_MRP: Maximum Retail Price of the product (List Price).
       - Outlet_Identifier: Unique identifier for each outlet.
       - Outlet_Establishment_Year: Year the outlet was established.
       - Outlet_Size: The Size of the store in terms of ground area covered.
       - Outlet_Location_Type: Type of city in which the store is located.
       - Outlet_Type: Outlet is grocery store or type of supermarket.
       - Item_Outlet_Sales: Target variable representing product sales.''')
       st.subheader('Data Set')
       st.dataframe(df.head())

       fig = px.histogram(df,x='Item_Outlet_Sales',nbins=50,title='Sales Distribution')
       st.plotly_chart(fig, use_container_width=True)

       fig = px.scatter(df,x="Item_MRP",y="Item_Outlet_Sales",title="Maximum Retail Price vs Sales")
       st.plotly_chart(fig, use_container_width=True)

       sales_by_outlet = (df.groupby("Outlet_Location_Type")["Item_Outlet_Sales"].mean().reset_index())
       fig = px.bar(sales_by_outlet,x="Outlet_Location_Type",y="Item_Outlet_Sales", text_auto=True, title="Average Sales by Outlet_Location_Type").update_xaxes(categoryorder='total descending')
       st.plotly_chart(fig, use_container_width=True)

       sales_by_outlet = (df.groupby("Outlet_Identifier")["Item_Outlet_Sales"].mean().reset_index())
       fig = px.bar(sales_by_outlet,x="Outlet_Identifier",y="Item_Outlet_Sales", text_auto=True, title="Average Sales by Outlet Identifier").update_xaxes(categoryorder='total descending')
       st.plotly_chart(fig, use_container_width=True)

       outlet_counts = (df.groupby("Outlet_Type").size().reset_index(name="Count"))
       fig = px.pie(outlet_counts,names="Outlet_Type",values="Count",title="Outlet Type Distribution")
       st.plotly_chart(fig, use_container_width=True)

       cat_col = st.selectbox("Select Category",["Outlet_Type","Outlet_Size","Item_Fat_Content"])
       grouped = (df.groupby(cat_col)["Item_Outlet_Sales"].mean().reset_index())
       fig = px.bar(grouped,x=cat_col,y="Item_Outlet_Sales", text_auto=True, title=f"Average Sales by {cat_col}").update_xaxes(categoryorder='total descending')
       st.plotly_chart(fig, use_container_width=True)

       top_items = (df.groupby("Item_Type")["Item_Outlet_Sales"].mean().sort_values(ascending=False).head(10).reset_index())
       fig = px.bar(top_items,x="Item_Type",y="Item_Outlet_Sales", text_auto=True, title="Top 10 Item Types")
       st.plotly_chart(fig, use_container_width=True)

       visibility = (df.groupby("Item_Type")["Item_Visibility"].mean().reset_index())
       fig = px.bar(visibility,x="Item_Type",y="Item_Visibility", text_auto=True, title="Most Displayed Item Types").update_xaxes(categoryorder='total descending')
       st.plotly_chart(fig, use_container_width=True)

elif page == 'Prediction':

       st.title('Prediction')

       item_weight = st.number_input('Item Weight', min_value=0.00)
       item_visibility = st.number_input('Item Visibility', min_value=0.00, format='%3f')
       item_mrp = st.number_input('Item MRP', min_value=0.00)
       outlet_year = st.number_input('Outlet Establishment Year', min_value=1900)
       quantity = st.number_input('Quantity', min_value=0.00)
       item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])

       item_type = st.selectbox('Item Type', ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
              'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
              'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
              'Breads', 'Starchy Foods', 'Others', 'Seafood'])

       outlet_identifier = st.selectbox('Outlet Identifier', ['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045',
              'OUT017', 'OUT046', 'OUT035', 'OUT019'])

       outlet_size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])

       outlet_location = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])

       outlet_type = st.selectbox('Outlet Type', ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

       if st.button('Predict'):

              data = pd.DataFrame({
              'Item_Weight': [item_weight],
              'Item_Fat_Content': [item_fat_content],
              'Item_Visibility': [item_visibility],
              'Item_Type': [item_type],
              'Item_MRP': [item_mrp],
              'Outlet_Identifier': [outlet_identifier],
              'Outlet_Establishment_Year': [outlet_year],
              'Outlet_Size': [outlet_size],
              'Outlet_Location_Type': [outlet_location],
              'Outlet_Type': [outlet_type],
              'Quantity': [quantity]
              })

              prediction = loaded_model.predict(data)


              st.success(f'Predicted Sales: {prediction[0]:,.2f}')
              st.write(data)
