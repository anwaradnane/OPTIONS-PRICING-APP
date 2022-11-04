import streamlit as st
import base64
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import math
import scipy


st.title('OPTIONS PRICING APPLICATION')

st.markdown("""
l'application a comme objectif de faire le pricing des options en utilisant l'un des modeles suivants: 
* **Binomial
* **Trinomial
* **Black Schools
* **Monee Carlo
""")

st.sidebar.header('LA BARRE DES OPTIONS DES MODELES')
# Sidebar - Sector selection
df =['BINOMIAL','TRINOMIAL','BLACK SCHOOLES','MONTE CARLO']
MODELE_PRICING =df
selected_MODELE = st.sidebar.multiselect('MODELE MATHEMATIQUE CHOISI', MODELE_PRICING,MODELE_PRICING)
S0= st.sidebar.slider("Cours initial de l'action",20,200,100)
K = st.sidebar.slider("prix d'exercice",8,210,60)
T= st.sidebar.slider('délai de maturité ',1,100,22)
r = st.sidebar.slider('taux annuel sans risque',0,100,6)
N = st.sidebar.slider('numero des itterations', 1,100, 44)
sigma = st.sidebar.slider('sigma %',0,100,3)
b=['Call','Put']
opt=st.sidebar.selectbox('option type',b)

sigma=sigma*10**-2
r=r*10**-2


def user_input_features():
    data = {"Cours initial de l'action": S0,
            "Prix d'exercice": K,
            'Délai de maturité': T,
            'Taux annuel sans risque': r,
            'Numero des itterations':N,
            'VOLATILITE':sigma }

    features = pd.DataFrame(data, index=[0])
    return features
x= user_input_features()

st.subheader('LES PARAMETRES DU MODELE SONT ')
st.write(x)
st.header('RESULTATS DES MODELES APRES LE CHOIX DES PARAMETRES')

for i in range(len(selected_MODELE)):
    if selected_MODELE[i] == 'BINOMIAL':
        st.subheader('DANS LE CAS DU MODEL  BINOMIAL ')
        def binomial_model(K, T, S0, r, N,sigma, opt):
            m = 1
            dt = T / N
            # parametres
            u = np.exp(sigma * np.sqrt(dt))
            d = np.exp(-sigma * np.sqrt(dt))
            p = (np.exp(r * dt) - d) / (u - d)

            ######combinaison fct pour le model################
            def combos(n, i):
                return math.factorial(n) / (math.factorial(n - i) * math.factorial(i))

            C = 0
            P = 0
            for k in reversed(range(N + 1)):
                p_ = combos(N, k) * p ** k * (1 - p) ** (N - k)
                ST = S0 * u ** k * d ** (N - k)
                if opt == 'Call':
                    C += max(ST - K, 0) * p_
                else:
                    P += max(0, K - ST) * p_

            if opt == 'Call':
                return np.exp(-r * T) * C
            else:
                return np.exp(-r * T) * P
        st.write("AVEC LE MODEL BINOMIAL, LE PRIX DU CALL DEVRAIT ETRE ",binomial_model(K, T, S0, r, N,sigma, opt))
        S = np.arange(0.1,1.5,0.01)
        calls = [binomial_model(K, T, S0, r, N, s, 'Call') for s in S]
        puts = [binomial_model(K, T, S0, r, N, s, 'Put') for s in S]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(S, calls, label='Call')
        ax.scatter(S, puts, label='Put')
        ax.set_xlabel('volatilite')
        ax.set_ylabel("la valeur de l'option")
        st.pyplot(fig)
    elif selected_MODELE[i] == 'TRINOMIAL':
        st.subheader('DANS LE CAS DU MODEL  TRINOMIAL ')
        def trinomial_Modele(K, T, S0, r, N,sigma, opt):
            q = 0
            m = 1
            dt = T / N
            u = np.exp(sigma * np.sqrt(2 * dt))
            d = 1 / u
            pu = ((math.exp((r - q) * dt / 2) - math.exp(-sigma * math.sqrt(dt / 2))) /
                  (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2)))) ** 2
            pd = ((math.exp(sigma * math.sqrt(dt / 2)) - math.exp((r - q) * dt / 2)) /
                  (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2)))) ** 2
            pm = 1 - pu - pd

            def ss(N):
                S = [np.array([S0])]
                for i in range(N):
                    prev_nodes = S[-1]  # At each loop, take the last element of the list
                    ST = np.concatenate((prev_nodes * u, [prev_nodes[-1] * m, prev_nodes[-1] * d]))
                    S.append(ST)
                return S

            call = [None] * N  # Empty list of size N
            put = [None] * N  # Empty list of size N
            for i in reversed(range(N)):
                if (i == N - 1):
                    payoffCall = np.maximum(0, max(ss(N)[i] - K))  # The payoff list is sorted, so take the first one
                    payoffPut = np.maximum(0, max((K - ss(N)[i])))  # The payoff list is sorted, so take the first one
                else:
                    payoffCall = math.exp(-(r - q) * dt) * (pu * call[i + 1] + pd * call[i + 1] + pm * call[i + 1])
                    payoffPut = math.exp(-(r - q) * dt) * (pu * put[i + 1] + pd * put[i + 1] + pm * put[i + 1])

                call.insert(i, (payoffCall))
                put.insert(i, (payoffPut))

            if opt == 'Call':
                return call[0]
            else:
                return put[0]
        st.write("AVEC LE MODEL TRINOMIAL, LE PRIX DU CALL DEVRAIT ETRE ",trinomial_Modele(K, T, S0, r, N,sigma, opt))
        S = np.arange(0.1, 1.5, 0.01)
        calls = [trinomial_Modele(K, T, S0, r, N, s, 'Call') for s in S]
        puts = [trinomial_Modele(K, T, S0, r, N, s, 'Put') for s in S]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(S, calls, label='Call')
        ax.scatter(S, puts, label='Put')
        ax.set_xlabel('volatilite')
        ax.set_ylabel("la valeur de l'option")
        st.pyplot(fig)

    elif selected_MODELE[i]=='BLACK SCHOOLES':
        st.subheader('DANS LE CAS DU MODEL DE BLACK SCHOOLES  ')
        def BS_(K, T, S, r,sigma, opt):
            N = norm.cdf
            if opt == 'Call':
                d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                return S * N(d1) - K * np.exp(-r * T) * N(d2)
            else:
                d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                return K * np.exp(-r * T) * N(-d2) - S * N(-d1)
        st.write("AVEC LE MODEL BLACK SCHOOELS, LE PRIX DU CALL DEVRAIT ETRE ",BS_(K, T, S0, r,sigma, opt))
        S = np.arange(0.1, 1.5, 0.01)
        calls = [BS_(K, T, S0, r,s, 'Call') for s in S]
        puts = [BS_(K, T, S0, r,s, 'Put') for s in S]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(S, calls, label='Call')
        ax.scatter(S, puts, label='Put')
        ax.set_xlabel('volatilite')
        ax.set_ylabel("la valeur de l'option")
        st.pyplot(fig)

    elif selected_MODELE[i]=='MONTE CARLO':
        st.subheader('DANS LE CAS DU MODEL  Monte Carlo ')
        def Monte_Carlo(K, T, S0, r,sigma, opt):
            numOfPath = 10000
            randomSeries = np.random.randn(numOfPath)
            s_t = S0 * np.exp((r - 0.5 * sigma * sigma) * T + randomSeries * sigma * math.sqrt(T))
            if opt == 'Call':
                sumValue = np.maximum(s_t - K, 0.0).sum()
                price = np.exp(-r * T) * sumValue / numOfPath
                return price
            else:
                sumValue = np.maximum(K - s_t, 0.0).sum()
                price = np.exp(-r * T) * sumValue / numOfPath
                return price


        st.write("AVEC LE MODEL MONTE CARLO, LE PRIX DU CALL DEVRAIT ETRE ", Monte_Carlo(K, T, S0, r,sigma, opt))

        S = np.arange(0.1, 1.5,0.01)
        calls = [Monte_Carlo(K, T, S0, r, s, 'Call') for s in S]
        puts = [Monte_Carlo(K, T, S0, r, s, 'Put') for s in S]
        fig, ax = plt.subplots(1, 1)
        ax.scatter(S, calls, label='Call')
        ax.scatter(S, puts, label='Put')
        ax.set_xlabel('volatilite')
        ax.set_ylabel("la valeur de l'option")
        st.pyplot(fig)

    else:
        print('PAS DU MODEL')





