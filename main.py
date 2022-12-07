import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import scipy
from scipy.optimize import minimize
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve, NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols

col_names = ["Date", "13 semaines", "26 semaines","52 semaines","2 ans","5 ans","10 ans","15 ans","25 ans","30 ans"]
df = pd.read_csv('BDT-2000-2022.csv',names=col_names,delimiter=',', skiprows=0, low_memory=False)
df=pd.DataFrame(df)
Maturite=[13,26,52,104,282,484,766,1300,1512]
st.title('COURBE DES TAUX APPLICATION')
st.subheader('LA BASE DES DONNEES EST ')
st.write(df)
st.subheader('MODELES DETERMINISTES')
DETERMINISTE =['interpolation linaire','interpolation Cubique','Nelson Siegel','Nelson Siegle Svensson']
selected_DET = st.sidebar.multiselect('MODELE DETERMINISTE CHOISI', DETERMINISTE,DETERMINISTE)
DATE_ = st.selectbox('DATE CHOISI',df.Date)
u = df.loc[df['Date'] == DATE_]
u = u.values.tolist()
arr = np.array(u)
arr = np.delete(arr, 0)
q = arr.tolist()
d = []
k = 0
for i in q:
    if i != '-':
        c = str(i).replace("%", "")
        d.append(float(c))

    else:

        k = k + 1
        d.append(float(c) + k * 0.3)
xnew = np.linspace(13,1512, num=28, endpoint=True)
beta0   = 0.1 # initial guess
beta1   = 0.1 # initial guess
beta2   = 0.1 # initial guess
beta3   = 0.5 # initial guess
lambda0 = 2 # initial guess
lambda1 = 5 # initial guess
from scipy.interpolate import interp1d
def interpolantion_lineaire(x, y):

            def interpFn(x0):
                if x0 < x[0] or x0 > x[-1]:
                    raise BaseException
                elif x0 == x[0]:
                    return y[0]
                elif x0 == x[-1]:
                    return y[-1]
                else:
                    i2 = 0
                    while x0 > x[i2]:
                        i2 += 1
                    i1 = i2 - 1
                    t = (x0 - x[i1]) / (x[i2] - x[i1])
                    return y[i1] * (1 - t) + t * y[i2]

            return interpFn
f = interp1d(Maturite, d)
from scipy.interpolate import CubicSpline
f2 =CubicSpline(Maturite,d)
for i in range(len(selected_DET)):
    if selected_DET[i] == 'interpolation linaire':
        st.subheader("Dans le cas du modele d'interpolation linaire" )
        from scipy.interpolate import interp1d
        def interpolantion_lineaire(x, y):

            def interpFn(x0):
                if x0 < x[0] or x0 > x[-1]:
                    raise BaseException
                elif x0 == x[0]:
                    return y[0]
                elif x0 == x[-1]:
                    return y[-1]
                else:
                    i2 = 0
                    while x0 > x[i2]:
                        i2 += 1
                    i1 = i2 - 1
                    t = (x0 - x[i1]) / (x[i2] - x[i1])
                    return y[i1] * (1 - t) + t * y[i2]

            return interpFn
        f = interp1d(Maturite, d)
        plt.plot(Maturite, d, 'o', xnew, f(xnew), '-')
        plt.legend(['data', 'linear'], loc='best')
        plt.show()
    elif selected_DET[i] == 'interpolation Cubique':
        def interpolation_cubique(x, y):
            def interpFn(xx):
                if xx <= x[1] or xx > x[-2]:
                    return float('nan')
                else:
                    i2 = 0
                    while xx > x[i2]:
                        i2 += 1
                    i1 = i2 - 1
                    i0 = i1 - 1;
                    i3 = i2 + 1;

                    x0 = x[i0]
                    x1 = x[i1]
                    x2 = x[i2]
                    x3 = x[i3]
                    y0 = y[i0]
                    y1 = y[i1]
                    y2 = y[i2]
                    y3 = y[i3]

                    d0 = (y1 - y0) / (x1 - x0)
                    h0 = (x1 - x0)
                    d1 = (y2 - y1) / (x2 - x1)
                    h1 = (x2 - x1)
                    d2 = (y3 - y2) / (x3 - x2)
                    h2 = (x3 - x2)

                    if d0 * d1 > 0:
                        w01 = h0 + 2 * h1
                        w11 = h1 + 2 * h0
                        m1 = (w01 + w11) / (w01 / d0 + w11 / d1)
                    else:
                        m1 = 0

                    if d1 * d2 > 0:
                        w02 = h1 + 2 * h2
                        w12 = h2 + 2 * h1
                        m2 = (w02 + w12) / (w02 / d1 + w12 / d2)
                    else:
                        m2 = 0

                    p1 = y1
                    p2 = y2

                    t = (xx - x1) / (x2 - x1)
                    res1 = (2 * (t ** 3) - 3 * (t ** 2) + 1) * p1
                    res2 = (t ** 3 - 2 * (t ** 2) + t) * (x2 - x1) * m1
                    res3 = (-2 * (t ** 3) + 3 * (t ** 2)) * p2
                    res4 = ((t ** 3) - (t ** 2)) * (x2 - x1) * m2
                    res = res1 + res2 + res3 + res4
                    return res

            return interpFn


        from scipy.interpolate import CubicSpline

        f2 = CubicSpline(Maturite, d)
        plt.plot(Maturite, d, 'o', xnew, f2(xnew), '-')
        plt.legend(['data', 'Cubique'], loc='best')
        plt.show()
    elif selected_DET[i] == 'Nelson Siegel':


        def NelsonSiegel(T, beta0, beta1, beta2, lambda0):
            alpha1 = (1 - np.exp(-T / lambda0)) / (T / lambda0)
            alpha2 = alpha1 - np.exp(-T / lambda0)
            return beta0 + beta1 * alpha1 + beta2 * alpha2


        def NSGoodFit(params, TimeVec, YieldVec):
            return np.sum((NelsonSiegel(TimeVec, params[0], params[1], params[2], params[3]) - YieldVec) ** 2)


        def NSMinimize(beta0, beta1, beta2, lambda0, TimeVec, YieldVec):
            optT_sol = minimize(NSGoodFit, x0=np.array([beta0, beta1, beta2, lambda0]), args=(TimeVec, YieldVec))
            if (optT_sol.success):
                return optT_sol.x
            else:
                return []


        TimeVec = np.array(Maturite)
        YieldVec = np.array(d)
        ## Implementation
        kk = NSMinimize(beta0, beta1, beta2, lambda0, TimeVec, YieldVec)
        print(NelsonSiegel(xnew, kk[0], kk[1], kk[2], kk[3]))
        plt.plot(Maturite, d, 'o', xnew,NelsonSiegel(xnew, kk[0], kk[1], kk[2], kk[3]),'-')
        plt.legend(['data', 'NelsonSiegel'], loc='best')
        plt.show()

    elif selected_DET[i] =='Nelson Siegle Svensson':
        def NelsonSiegelSvansson(T, beta0, beta1, beta2, beta3, lambda0, lambda1):
            alpha1 = (1 - np.exp(-T / lambda0)) / (T / lambda0)
            alpha2 = alpha1 - np.exp(-T / lambda0)
            alpha3 = (1 - np.exp(-T / lambda1)) / ((T / lambda1) - np.exp(-T / lambda1))
            return beta0 + beta1 * alpha1 + beta2 * alpha2 + beta3 * alpha3


        def NSSGoodFit(params, TimeVec, YieldVec):
            return np.sum((NelsonSiegelSvansson(TimeVec, params[0], params[1], params[2], params[3], params[4],params[5]) - YieldVec) ** 2)


        def NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, TimeVec, YieldVec):
            opt_sol = minimize(NSSGoodFit, x0=np.array([beta0, beta1, beta2, beta3, lambda0, lambda1]),args=(TimeVec, YieldVec), method="Nelder-Mead")
            if (opt_sol.success):
                return opt_sol.x
            else:
                return []


        TimeVec = np.array(Maturite)
        YieldVec = np.array(d)
        kk = NSSMinimize(beta0, beta1, beta2, beta3, lambda0, lambda1, TimeVec, YieldVec)
        print(NelsonSiegelSvansson(xnew,kk[0], kk[1], kk[2], kk[3], kk[4], kk[5]))
        plt.plot(Maturite, d, 'o', xnew, NelsonSiegelSvansson(xnew, kk[0], kk[1], kk[2], kk[3], kk[4], kk[5]),'-')
        plt.legend(['data', 'NelsonSiegel Svansson'], loc='best')
        plt.show()
fig, ax= plt.subplots(1, 1)
curve_fit1, status1 = calibrate_ns_ols(np.array(Maturite),np.array(d)) #NS model calibrate
curve_fit, status = calibrate_nss_ols(np.array(Maturite),np.array(d)) #NSS model calibrate
NS_ZC = NelsonSiegelCurve.zero(curve_fit1,np.array(xnew))
NSS_ZC = NelsonSiegelSvenssonCurve.zero(curve_fit,np.array(xnew))
ax.plot(xnew,f(xnew), label='LINEAIRE')
ax.plot(xnew,f2(xnew), label='CUBIQUE')
ax.plot(xnew,NS_ZC,label='NS')
ax.plot(xnew,NSS_ZC,label='NSS')
ax.set_xlabel('semaines')
ax.set_ylabel("taux d'interet")
st.pyplot(fig)









