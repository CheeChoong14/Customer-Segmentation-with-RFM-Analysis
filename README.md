# Customer Segmentation with RFM Analysis
This project demonstrates a complete end-to-end workflow for customer segmentation using RFM (Recency, Frequency, Monetary) analysis and unsupervised machine learning techniques.

## Dataset Description

- **File**: `orders.parquet`
- **Columns**:
  - `id`: Unique order ID
  - `created_at`: Timestamp of the order
  - `sales_amount`: Order value in MYR
  - `customer_id`: Unique customer identifier

## Objectives

- Perform data cleaning and quality checks
- Conduct exploratory data analysis (EDA)
- Engineer RFM features at customer level
- Cluster customers with unsupervised learning (MiniBatch KMeans)
- Generate business insights and strategies per segment


## Tools & Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `plotly`
- `scikit-learn` (MiniBatchKMeans, silhouette_score)
- `datetime`, `warnings`, `resample`

## Project Workflow

### 1. Data Loading & Cleaning
- Loaded data from `.parquet` format
- Checked for missing values and data types
- Parsed `created_at` to datetime and sorted records

### 2. Exploratory Data Analysis (EDA)
- Visualized order volume by:
  - **Quarter** (`Q1–Q4`)
  - **Month** (line plot)
  - **Week**
  - **Day of Week** (Mon–Sun)
  - **Hour of Day** (peak hours)

### 3. RFM Feature Engineering
- Grouped transactions by `customer_id`:
  - **Recency**: Days since last purchase
  - **Frequency**: Number of orders
  - **Monetary**: Total spending
- Normalized features for clustering

### 4. Customer Segmentation (Clustering)
- Used **MiniBatchKMeans** for efficient clustering
- Determined best number of clusters using:
  - **Elbow Method (SSE)**
  - **Silhouette Score**
- Labeled clusters as:
  - `Champions`
  - `Loyal Customers`
  - `Normal Customers`

### 5. Cluster Interpretation
- Analyzed RFM distribution per cluster
- Created segment-level visualizations and summaries
- Proposed business strategies per segment


## How to Run

```bash
# Install necessary packages
pip install pandas matplotlib seaborn scikit-learn plotly

# Launch Jupyter
jupyter notebook test.ipynb
