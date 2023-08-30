#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[3]:


data = pd.read_csv(r"C:\Users\Mhd Naqeeb\Downloads\demand_inventory.csv")
print(data.head())


# In[4]:


data = data.drop(columns=['Unnamed: 0'])


# In[5]:


fig_demand = px.line(data, x='Date',
                     y='Demand',
                     title='Demand Over Time')
fig_demand.show()


# In[6]:


fig_inventory = px.line(data, x='Date',
                        y='Inventory',
                        title='Inventory Over Time')
fig_inventory.show()


# Demand Forecasting:
# forecast the demand using SARIMA.calculateing the value of p and q using ACF and PACF plots:

# In[8]:


#Adjust the Date Format
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')


# In[9]:


data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
time_series = data.set_index('Date')['Demand']

differenced_series = time_series.diff().dropna()


# In[10]:


# Plot ACF and PACF of differenced time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(differenced_series, ax=axes[0])
plot_pacf(differenced_series, ax=axes[1])
plt.show()


# In[11]:


order = (1, 1, 1)
seasonal_order = (1, 1, 1, 2) #2 because the data contains a time period of 2 months only
model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

future_steps = 10
predictions = model_fit.predict(len(time_series), len(time_series) + future_steps - 1)
predictions = predictions.astype(int)
print(predictions)


# Inventory Optimization

# In[12]:


# Create date indices for the future predictions
future_dates = pd.date_range(start=time_series.index[-1] + pd.DateOffset(days=1), periods=future_steps, freq='D')


# In[13]:


# Create a pandas Series with the predicted values and date indices
forecasted_demand = pd.Series(predictions, index=future_dates)


# In[14]:


# Initial inventory level
initial_inventory = 5500


# In[15]:


# Lead time (number of days it takes to replenish inventory) 
lead_time = 1 
# Service level (probability of not stocking out)
service_level = 0.95 


# In[16]:


# Calculate the optimal order quantity using the Newsvendor formula
z = np.abs(np.percentile(forecasted_demand, 100 * (1 - service_level)))
order_quantity = np.ceil(forecasted_demand.mean() + z).astype(int)

# Calculate the reorder point
reorder_point = forecasted_demand.mean() * lead_time + z


# In[17]:


# Calculate the optimal safety stock
safety_stock = reorder_point - forecasted_demand.mean() * lead_time

# Calculate the total cost (holding cost + stockout cost)
holding_cost = 0.1  # it's different for every business, 0.1 is an example
stockout_cost = 10  # # it's different for every business, 10 is an example
total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_stockout_cost = stockout_cost * np.maximum(0, forecasted_demand.mean() * lead_time - initial_inventory)


# In[18]:


# Calculate the total cost
total_cost = total_holding_cost + total_stockout_cost

print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)


# In[ ]:


By analyzing these values,this is how decisions about inventory to order and when to place orders to ensure a smooth supply chain and customer satisfaction while minimizing costs.

