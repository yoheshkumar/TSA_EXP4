### DEVELOPED BY: YOHESHKUMAR R.M
### REGISTER NO: 212222240118
### DATE:


# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# AIM:
To implement ARMA model in python.
# ALGORITHM:
1. Import necessary libraries such as `numpy`, `matplotlib`, `pandas`, and `statsmodels`.
2. Load the dataset using `pandas.read_csv()`.
3. Extract the relevant column from the dataset (e.g., `DAYTON_MW`).
4. Define the ARMA(1,1) process with AR(1) and MA(1) coefficients.
5. Generate a sample of 1000 data points using the `ArmaProcess` class for ARMA(1,1).
6. Plot the generated ARMA(1,1) time series using `matplotlib`.
7. Display the ACF and PACF for the ARMA(1,1) process using `plot_acf()` and `plot_pacf()`.
8. Define the ARMA(2,2) process with AR(2) and MA(2) coefficients.
9. Generate a sample of 10,000 data points for the ARMA(2,2) process.
10. Plot the generated ARMA(2,2) time series.
11. Display the ACF and PACF for the ARMA(2,2) process.
# PROGRAM:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the Microsoft stock dataset
file_path = 'Microsoft_Stock.csv'  # Replace this with your file path if different
data = pd.read_csv(file_path)

# Extract the 'Close' price column for modeling
data_values = data['Close'].dropna().values

# 1. ARMA(1,1) Process for Microsoft Stock Data

# Define AR and MA coefficients for ARMA(1,1)
ar1 = np.array([1, -0.5])  # AR(1) coefficient
ma1 = np.array([1, 0.5])   # MA(1) coefficient

# Generate a sample based on the length of Microsoft stock data
arma11_process = ArmaProcess(ar1, ma1)
arma11_sample = arma11_process.generate_sample(nsample=len(data_values))

# Plot the ARMA(1,1) time series for Microsoft stock data
plt.figure(figsize=(10, 6))
plt.plot(arma11_sample)
plt.title('Generated ARMA(1,1) Process for Microsoft Stock')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim(0, len(arma11_sample))
plt.grid(True)
plt.show()

# Display ACF and PACF plots for ARMA(1,1)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plot_acf(arma11_sample, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(arma11_sample, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for ARMA(1,1) - Microsoft Stock')
plt.tight_layout()
plt.show()

# 2. ARMA(2,2) Process for Microsoft Stock Data

# Define AR and MA coefficients for ARMA(2,2)
ar2 = np.array([1, -0.7, 0.3])  # AR(2) coefficients
ma2 = np.array([1, 0.5, 0.4])   # MA(2) coefficients

# Generate a sample based on the length of Microsoft stock data
arma22_process = ArmaProcess(ar2, ma2)
arma22_sample = arma22_process.generate_sample(nsample=len(data_values))

# Plot the ARMA(2,2) time series for Microsoft stock data
plt.figure(figsize=(10, 4))
plt.plot(arma22_sample)
plt.title('Generated ARMA(2,2) Process for Microsoft Stock')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim(0, len(arma22_sample))
plt.grid(True)
plt.show()

# Display ACF and PACF plots for ARMA(2,2)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plot_acf(arma22_sample, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(arma22_sample, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for ARMA(2,2) - Microsoft Stock')
plt.tight_layout()
plt.show()


```


# OUTPUT:
## SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/user-attachments/assets/6def2f73-010a-4da1-8fba-fdf7b0a015b8)


## Partial Autocorrelation
![image](https://github.com/user-attachments/assets/c9cfd0d0-7d92-41a0-907f-4f85febaa6f5)


## Autocorrelation
![image](https://github.com/user-attachments/assets/79e3edbf-60dc-4701-91cd-27a0a1d3557b)



## SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/69cef9d7-398e-4448-bf0b-c4a3d64b3861)


## Partial Autocorrelation
![image](https://github.com/user-attachments/assets/457ff424-7208-465c-8878-68927c86686b)



## Autocorrelation
![image](https://github.com/user-attachments/assets/0fd608c0-844b-4e05-8a90-c5949068c67f)



# RESULT:
Thus, a python program is created to fit ARMA Model for Time Series successfully.
