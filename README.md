## Overview

This project involves analyzing data and developing a Streamlit dashboard to visualize key business metrics for the ChipChip social buying platform.

### About ChipChip: Our Social Buying Platform

ChipChip is a unique marketplace that connects sellers and customers by enabling users to make purchases either individually or as part of a group, enjoying discounted prices. Groups have a limited time (e.g., 24 hours, 48 hours..etc) to reach their member count goal (e.g., 5/5, 3/3). If the goal isn’t met, the group is considered expired and refunds are processed. This mechanism encourages community engagement and savings through collective purchasing.

## Directory Structure

The directory structure for the project is as follows:


- [notebooks]: Jupyter notebooks for data analysis and exploration.
- [streamlit_app]: Contains the Streamlit application code.
  - `app.py`: Main Streamlit application file.
  - `preprocessing/`: Contains preprocessing scripts.
    - `data_preprocessor.py`: Script for data preprocessing.
- [.gitignore]: Specifies files and directories to be ignored by Git.
- [ChipChip_Report.pdf]: Project Report.
- [README.md]: Project documentation.
- [requirements.txt]: List of dependencies required for the project.
- [Struture.md]: Detailed structure of the database.


## Features

- **Dynamic Heatmap**: Visualize the correlation between product categories and order contributions across vendors.
- **Time-Series Forecast**: Forecast order trends using ARIMA and Prophet models.
- **Grouped Bar Chart**: Compare order quantities for group deals vs. individual deals.
- **Performance Metrics**: Compare performance metrics (e.g., revenue, conversion rates, user retention) for different product categories.
- **User Segmentation**: Segment users into different groups (e.g., high-value customers, occasional buyers) using K-Means clustering.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Anaconda (recommended for managing dependencies)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/chipchip-dashboard.git
   cd chipchip-dashboard
   ```

2. **Create and activate a virtual environment**:

   Using Anaconda:
   ```bash
   conda create --name chipchip-dashboard python=3.8
   conda activate chipchip-dashboard
   ```

   Using venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run streamlit_app/app.py
   ```

2. **Open the app in your browser**:

   The app will be available at `http://localhost:8501`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [Streamlit](https://www.streamlit.io/)
- [Plotly](https://plotly.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Prophet](https://facebook.github.io/prophet/)
- [scikit-learn](https://scikit-learn.org/)
```