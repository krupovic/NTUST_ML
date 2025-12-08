# Centralized imports for the NTUST_ML project
# Import this file in other scripts as: from project_imports import *

# File handling
import zipfile

# Data processing
import pandas as pd
import datetime
import numpy as np
import re

# Visualization
import folium
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine learning
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
