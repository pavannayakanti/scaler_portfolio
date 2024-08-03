# Sales Forecasting and Analysis Dashboard
#### [View the Sales Performance Dashboard](https://public.tableau.com/app/profile/pavan.kumar.nayakanti/viz/Scaler-Portfolio-ProductSalesForecast/1_SalesPerformanceDashboard)

## Problem Statement

In the competitive retail industry, the ability to predict future sales accurately is crucial for operational and strategic planning...

## Need and Use of Product Sales Forecasting

### Inventory Management

Accurate sales forecasts help ensure...

### Financial Planning

Forecasting sales allows businesses to estimate future revenue...

## Tableau Visualizations and Dashboards

### 1. Sales Performance Dashboard

#### Time Series Analysis
Line charts showing daily, weekly, or monthly sales trends...
Observation: The time series line chart shows noticeable seasonal trends and periodic peaks in sales, particularly during certain months, indicating seasonality.
Inference: Retailers experience higher sales volumes during specific periods, possibly due to seasonal promotions, holidays, or annual events. This insight can help in planning inventory and marketing strategies to capitalize on high-demand periods.

#### 2. Comparison by Store Type and Location
Bar charts or pie charts to display sales distribution...
Observation: The bar charts and pie charts reveal that certain store types and locations consistently outperform others in terms of sales volume.
Inference: The performance variation suggests that store characteristics and location significantly influence sales. Stores in urban or high-traffic areas (e.g., location type L1 and L2) tend to generate higher sales compared to those in less populated regions (e.g., L4 and L5). This insight can guide strategic decisions on store placement and resource allocation.

#### 3. Regional Sales Analysis
Observation: The comparison of regions shows that Region R1 has the highest sales volume, followed by R2, R3, and R4. Additionally, average order sizes vary significantly between regions.
Inference: Region-specific factors, such as demographic differences, regional preferences, and economic conditions, affect sales. Understanding these factors can help tailor marketing strategies and product assortments to suit regional tastes.

#### 4. Promotional Impact Analysis
Observation: The scatter plot for discount effectiveness indicates that sales tend to increase on days when discounts are offered. The box plot shows that the median sales on discount days are higher compared to non-discount days.
Inference: Discounts and promotions are effective in boosting sales. Retailers can leverage this by strategically planning discount events, especially during off-peak times, to stimulate demand and manage inventory levels.

#### 5. Holiday Sales Impact
Observation: The bar chart comparing sales on holidays versus regular days shows a significant increase in sales during holidays.
Inference: Holidays are key periods for retail sales spikes, driven by consumer behavior and holiday-specific promotions. Retailers should ensure adequate stock levels and staffing during these times to meet increased demand and maximize revenue.

#### 6. Daily Orders versus Sales
Observation: The scatter plot demonstrates a strong positive correlation between the number of orders and sales, indicating that more orders generally lead to higher sales figures.
Inference: Operational efficiency, such as minimizing order processing time and ensuring stock availability, is crucial for maintaining high sales. Any bottlenecks in order processing could directly impact sales performance.

#### 7. Stock Management Insights
Observation: The chart displaying daily sales changes indicates periods of significant fluctuation, which could point to potential overstocking or stock shortages.
Inference: Rapid changes in sales figures may indicate inefficiencies in inventory management, such as overstocking before anticipated demand increases or understocking during high demand periods. Accurate forecasting and responsive inventory management are vital for minimizing losses and optimizing stock levels.

#### 8. Forecast vs. Actual Sales
Observation: The line charts comparing forecasted and actual sales reveal discrepancies, with some periods showing close alignment and others significant divergence.
Inference: The accuracy of the forecasting model varies, and there may be external factors not captured by the model that affect sales. Continuous model refinement and the inclusion of additional influencing factors can improve forecast accuracy.


### Suggestions for Hypothesis Testing

### Analyzing Sales Data: Impact of Discounts, Holidays, Store Types, and Regions

In this technical blog, we explore the effects of various factors on sales using a dataset that includes information about discounts, holidays, store types, and regions. Our objective is to test the following hypotheses:

1. **Impact of Discounts on Sales**
2. **Effect of Holidays on Sales**
3. **Sales Differences Across Store Types**
4. **Regional Sales Variability**
5. **Correlation between Number of Orders and Sales**

#### 1. Impact of Discounts on Sales

**Hypothesis**: Stores offering discounts will have significantly higher sales than stores not offering discounts.

- **Test**: Independent t-test
- **Results**:
  - **t-statistic**: 148.58
  - **p-value**: 0.0

The results indicate a significant difference in sales between days with and without discounts, suggesting that discounts positively impact sales.

#### 2. Effect of Holidays on Sales

**Hypothesis**: Sales on holidays are higher compared to non-holidays.

- **Test**: Independent t-test
- **Results**:
  - **t-statistic**: -67.99
  - **p-value**: 0.0

Sales on holidays are significantly different from sales on non-holidays, which indicates that holidays have a substantial impact on sales.

#### 3. Sales Differences Across Store Types

**Hypothesis**: Different store types experience different sales volumes.

- **Test**: ANOVA
- **Results**:
  - **F-statistic**: 35123.64
  - **p-value**: 0.0

The ANOVA test results show a significant difference in sales across different store types, suggesting that store type influences sales volume.

#### 4. Regional Sales Variability

**Hypothesis**: There is significant variability in sales across different regions.

- **Test**: Kruskal-Wallis test
- **Results**:
  - **Statistic**: 3968.06
  - **p-value**: 0.0

The Kruskal-Wallis test results indicate significant variability in sales across regions, implying that sales performance varies significantly by region.

#### 5. Correlation between Number of Orders and Sales

**Hypothesis**: A higher number of orders correlates with higher sales.

- **Test**: Pearson correlation coefficient
- **Results**:
  - **Correlation coefficient**: 0.9416
  - **p-value**: 0.0

There is a strong positive correlation between the number of orders and sales, indicating that as the number of orders increases, sales also tend to increase.

### Conclusion

The analysis provides valuable insights into the factors influencing sales. Discounts, holidays, store types, and regions all significantly impact sales, and there is a strong correlation between the number of orders and sales. These findings can help businesses tailor their strategies to maximize sales based on these factors.

Reference: For further details and a practical implementation, refer to the [Product Sales Forecasting notebook](https://github.com/pavannayakanti/scaler_portfolio/blob/main/Others/Product%20sales%20forecasting.ipynb%20-%20Colab.pdf).


## Technical Implementation

### Data Collection and Preprocessing

#### 1. Data Collection
The dataset used for this project, `TRAIN.csv`, includes historical sales data, store details, and other relevant information. The data was collected from various retail stores and includes attributes such as sales volume, number of orders, discounts, holidays, and store types.

#### 2. Data Cleaning
To ensure the quality of the data, the following steps were undertaken:
- **Handling Missing Values:** Identified and imputed missing values using appropriate strategies such as filling with median or mode values for numerical and categorical data, respectively.
- **Removing Duplicates:** Checked and removed any duplicate records to prevent skewed analysis.
- **Date Parsing:** Converted date columns to datetime format for time series analysis.

#### 3. Feature Engineering
Feature engineering was performed to create new variables that could help improve the model's predictive power:
- **Time-Based Features:** Extracted features such as year, month, week, and day of the week.
- **Lag Features:** Created lag features such as sales in the previous week to capture temporal dependencies.
- **Categorical Encoding:** Applied one-hot encoding to categorical variables like store type, location type, and region code.

### Exploratory Data Analysis (EDA)

#### Univariate Analysis
- **Sales Distribution:** Histograms and box plots were used to study the distribution of sales and identify the presence of outliers.
- **Number of Orders:** Similar analysis was performed on the number of orders to understand its distribution.

#### Bivariate Analysis
- **Discount Impact:** Scatter plots were used to examine the relationship between discounts and sales, revealing that discounts generally lead to higher sales.
- **Holiday Effect:** Analyzed sales variations between holidays and non-holidays, confirming that sales tend to increase during holidays.
- **Store Type Analysis:** Compared sales across different store types, identifying which store types perform better.

### Hypothesis Testing

Several hypotheses were tested using statistical tests to validate assumptions:
1. **Discount vs. No Discount Sales:** A t-test was performed to check if sales on discount days differ significantly from non-discount days.
2. **Holiday vs. Non-Holiday Sales:** Another t-test was conducted to examine differences in sales during holidays.
3. **Store Type Sales Comparison:** An ANOVA test was used to compare sales across different store types.
4. **Regional Sales Differences:** The Kruskal-Wallis test was applied to identify significant differences in sales across regions.
5. **Correlation Analysis:** Pearson or Spearman correlation coefficients were calculated to measure the strength of the relationship between the number of orders and sales.

### Time Series Analysis

To analyze sales trends over time and make future sales forecasts:
- **Decomposition:** Applied Seasonal-Trend decomposition using LOESS (STL) to identify and separate the seasonal, trend, and residual components of the time series data.
- **Stationarity Check:** Conducted the Augmented Dickey-Fuller test to check the stationarity of the sales time series data.
- **Modeling:** Implemented ARIMA and SARIMA models to forecast future sales, with a focus on capturing both seasonal and non-seasonal components.

### Categorical Data Analysis

- **Sales by Store Type:** Bar charts illustrated total sales by store type, revealing which types of stores generated the most revenue.
- **Sales by Location Type:** Analyzed total sales across different location types to understand geographical performance.
- **Sales by Region Code:** Compared sales across various regions to identify regional sales patterns.

### Model Evaluation

#### Model Performance Metrics
- **Mean Absolute Error (MAE):** Calculated to measure the average magnitude of errors in a set of predictions.
- **Root Mean Squared Error (RMSE):** Used to measure the differences between values predicted by the model and the actual values.

#### Results and Interpretation
The analysis revealed that certain store types and locations consistently outperform others. Discounts and holidays are effective in boosting sales. The time series models showed varying degrees of accuracy in forecasting sales, with seasonal factors playing a significant role.

## Dashboard and Visualizations

The project includes a comprehensive set of dashboards and visualizations, available on [Tableau](https://public.tableau.com/views/YourDashboardName/YourSheetName). These visualizations provide interactive insights into the data, allowing for a deeper understanding of sales patterns and drivers.

## Conclusion and Recommendations

The analysis highlights the importance of understanding various factors that influence sales, such as store type, location, promotions, and seasonal trends. Retailers can leverage these insights to optimize inventory, tailor marketing strategies, and improve overall business performance. Future work could involve refining the forecasting models, incorporating more external data (like economic indicators), and exploring advanced machine learning techniques for more accurate predictions.

## Access the Code and Data

The full implementation, including data preprocessing, EDA, modeling, and visualizations, is available in the GitHub repository: [Sales Forecasting Repository](https://github.com/yourusername/your-repository).

---

# Model Development

## Model Development

### Data Preprocessing

**Data Cleaning:**
The dataset was initially processed to handle missing values, remove duplicates, and correct inconsistencies. Outliers were identified and handled, ensuring the data quality for subsequent analyses and modeling. Numerical columns with missing values were imputed using median or mean values, while categorical columns were imputed using the mode.

**Feature Engineering:**
Additional features were engineered to enhance the predictive power of the models. These features included:
- **Temporal Features:** Year, month, week, day, and day of the week.
- **Lag Features:** Sales from the previous week (`Sales_Last_Week`).
- **Interaction Terms:** Combined store type and location (`Store_Location`).

**Data Transformation:**
Data normalization and encoding were performed to prepare the data for modeling. Numerical features were scaled using standardization, and categorical variables were encoded using one-hot encoding.

**Train-Test Split:**
The data was split into training and testing sets to evaluate the model's performance. The split was stratified based on the store type to ensure representation in both sets.

### Model Selection

#### Baseline Model
A simple **Linear Regression** model was initially implemented as a baseline to establish a reference point for model performance.

#### Complex Models
To improve accuracy, more sophisticated models were explored:

1. **Time Series Models:**
   - **ARIMA:** Applied to model and forecast time series data.
   - **SARIMA:** Extended ARIMA to handle seasonality in the data.
   - **Prophet:** A robust time series forecasting model that can handle missing data and outliers.

2. **Tree-Based Models:**
   - **Random Forests:** Used for its ability to handle nonlinear relationships and feature interactions.
   - **Gradient Boosting Machines (XGBoost, LightGBM):** Enhanced model accuracy by focusing on misclassified instances.

3. **Deep Learning Models:**
   - **LSTM (Long Short-Term Memory):** Leveraged for its ability to capture long-term dependencies in sequential data.

4. **Ensemble Techniques:**
   - Combined predictions from multiple models to improve robustness and accuracy.

### Model Evaluation

**Performance Metrics:**
The models were evaluated using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²). These metrics provided insights into the model's accuracy and predictive capabilities.

**Forecast Evaluation:**
- **Actual vs. Forecasted Sales:** Line charts were plotted to compare actual sales with forecasted values.
- **Error Analysis:** Graphs displaying MAE and RMSE over time helped identify periods of higher error rates and potential areas for model improvement.

## Key Insights and Findings

### Seasonal and Promotional Impact

The analyses revealed significant seasonality in sales, with noticeable peaks during holiday seasons and discount periods. Promotions and discounts were found to effectively boost sales, with a higher incidence of large sales spikes during these periods.

### Store and Regional Performance

There were distinct differences in performance across store types and regions. Store types S1 and S4 consistently outperformed others, indicating a need to analyze the factors contributing to their success. Region R1 showed the highest sales volume, suggesting regional preferences or demographic factors that could be leveraged for targeted marketing.

### Operational Insights

The analysis of daily orders versus sales highlighted the efficiency of order processing and inventory management. Rapid changes in daily sales figures indicated potential overstocking or stock shortages, necessitating a more dynamic inventory management approach.

## Future Work

### Model Refinement
Future work will focus on refining the models by incorporating additional external factors such as economic indicators, competitor data, and weather conditions. Further tuning of hyperparameters and exploring alternative modeling techniques will also be undertaken to enhance model accuracy.

### Integration with Business Systems
The developed forecasting models can be integrated into business systems for real-time sales forecasting. This integration will allow for more proactive decision-making, such as adjusting inventory levels, scheduling promotions, and optimizing staffing.

### Continuous Monitoring and Improvement
A system for continuous monitoring of model performance will be established. This will include regular updates to the model with new data and retraining as necessary to maintain accuracy over time.

## Accessing the Code and Dashboard

- **GitHub Repository:** The complete codebase, including data preprocessing, feature engineering, model training, and evaluation, is available in the GitHub repository. Access it [here](https://github.com/pavannayakanti/scaler_portfolio).
- **Tableau Dashboard:** Interactive dashboards for exploring the data visualizations and insights can be accessed [here](https://public.tableau.com/app/profile/pavan.kumar.nayakanti/viz/Scaler-Portfolio-ProductSalesForecast/1_SalesPerformanceDashboard).

---