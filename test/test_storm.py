import HurricaneForecast
from math import isclose
import pandas as pd
import numpy as np

# Set up the test dataframe
df_test = pd.DataFrame(np.array([['abc_001', 'abc', '437401', '1', '130'],
['abc_002', 'abc', '441000', '1', '140'],
['abc_003', 'abc', '500000', '1', '145'],
['abc_004', 'abc', '501800', '1', '100'],
['abc_005', 'abc', '503600', '1', '90']]),
columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean', 'Wind Speed'])
download_directory = 'download_dir'
train_source = 'train_source'
train_labels = 'train_labels'

df_test['Relative Time'] = df_test['Relative Time'].apply(int)
df_test['Wind Speed'] = df_test['Wind Speed'].apply(int)
download_directory = 'download_dir'
train_source = 'train_source'
train_labels = 'train_labels'


def test_check_time_gaps():
    """
    test the calculation of time gaps in Relative Time field
    """
    data_setup = HurricaneForecast.DataChecks(download_directory, train_source, train_labels, provide_df = True, df=df_test)
    data_setup.check_time_gaps()


    assert isclose(data_setup.times[0], 58999, abs_tol=1e+1)

def test_average_wind_speeds():
    """
    test the calculation of average wind speed
    """
    data_setup = HurricaneForecast.DataChecks(download_directory, train_source, train_labels, provide_df = True, df=df_test)
    avg_wind = data_setup.average_wind_speeds()

    print(avg_wind)

    assert isclose(avg_wind, 121, abs_tol=1e+1)

def test_count_images():
    """
    test the count of images
    """
    data_setup = HurricaneForecast.DataChecks(download_directory, train_source, train_labels, provide_df = True, df=df_test)
    num_img = data_setup.count_images()

    print(num_img)

    assert isclose(num_img, 5, abs_tol=1e+0)
