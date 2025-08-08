This personal project builds a urban analytics dashboard, transforming 311 service request data in Chicago into predictive insights.
By cleaning and transforming community-level data from the city of Chicago, I built a time-series model using a XGBoost regressor. 
Then, I funneled these predictions into an interactive Bokeh dashboard that provides granular detail. 

Below is a link to the web dashboard, updated daily and available 24/7:

## ❗[Click here to visit the forecasting dashboard](https://chicago-service-dashboard.onrender.com/app)❗


The following files are used to build the underlying backend and frontend of the dashboard:
- **[Chicago Servcie Requests Dashboard - Overview Report](https://github.com/dominickyacono/chicago-service-dashboard/blob/main/Chicago%20Servcie%20Requests%20Dashboard%20-%20Overview%20Report.pdf)**
  This PDF report gives an overview of the predictive model and the visual dashboard, walking through the developmental steps taken.
- **[Miscellaneous](https://github.com/dominickyacono/chicago-service-dashboard/main/miscellaneous)**
  This folder holds supporting project elements like editor settings, cached data, and software dependencies.
- **[Data Files](https://github.com/dominickyacono/chicago-service-dashboard/main/data_file)**
  This folder holds geographic mapping data, historical service request data, and predictions for service requests.
- **[The App Code](https://github.com/dominickyacono/chicago-service-dashboard/main/app.py)**
  The python code to build the Bokeh dashboard to show the service request predictions.
- **[The Data Pipeline Code ](https://github.com/dominickyacono/chicago-service-dashboard/main/pipeline.py)**
  This python code provides the step-by-step process of data importing, cleaning, feature engineering, model building, training, and testing.

  
  

