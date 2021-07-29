import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, chi2
import matplotlib.pyplot as plt
from numpy import pi, sin, cos
import plotly.graph_objects as go
import plotly.express as px


# To run the app type: streamlit run PLS_SL1.py

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The PLS App',
    layout='wide')

#---------------------------------#
# Model building


def ellipse(x_center=0, y_center=0, ax1 = [1, 0],  ax2 = [0,1], a=1, b =1,  N=360):
    # x_center, y_center the coordinates of ellipse center
    # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
    # a, b the ellipse parameters
    if np.linalg.norm(ax1) != 1 or np.linalg.norm(ax2) != 1:
        raise ValueError('ax1, ax2 must be unit vectors')
    if  abs(np.dot(ax1, ax2)) > 1e-06:
        raise ValueError('ax1, ax2 must be orthogonal vectors')
    t = np.linspace(0, 2*pi, N)
    #ellipse parameterization with respect to a system of axes of directions a1, a2
    xs = a * cos(t)
    ys = b * sin(t)
    #rotation matrix
    R = np.array([ax1, ax2]).T
    # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
    xp, yp = np.dot(R, [xs, ys])
    x = xp + x_center 
    y = yp + y_center
    return x, y


def build_model(df):
    data = df.to_numpy()
    X = data[:, 0:15]
    y = data[:, 15]
    N = X.shape[0]
    P = X.shape[1]

    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = y.reshape(-1,1)
    y = sc.fit_transform(y)

    A = 2
    pls2 = PLSRegression(n_components= A )
    pls2.fit(X, y)


    # variance of each latent value
    var = np.var(pls2.x_scores_, axis=0, ddof=1)

    # socres in X
    t = pls2.x_scores_

    T_hot = np.sum(np.square(t) / var, axis=1)
    crit_99 = f.ppf(0.99, A, N-A) * A * (N**2 - 1) / (N * (N - A)) 

    # Plotting the first two components
    a = np.sqrt(var[0] * crit_99)
    b = np.sqrt(var[1] * crit_99)

    fig = go.Figure()
    fig.add_hline(y=0)
    fig.add_vline(x=0)
    x, y = ellipse(a=a, b=b)
    fig.add_trace(go.Scatter(x=x, y=y, mode = 'lines', name='lim 99%'))

    fig.add_trace(go.Scatter(x=pls2.x_scores_[:,0], y=pls2.x_scores_[:,1], marker_color=df['cylindernumber'], marker_showscale=True, mode='markers', hovertemplate=df.index, name='Data'))

    fig.update_layout(width =800, height=600)
    st.pyplot(fig)



st.title('PLS App V1')

st.markdown("""
This app performs a PLS for the data selected. It can plot the components, T2 HOtelling and SPE to analyse the observations.
""")

st.sidebar.header('User Input Features')

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Variables'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)



#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.set_index("Car model", inplace=True)  # only needed for this test dataset
    st.markdown('**1.1. Components Plot**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')

