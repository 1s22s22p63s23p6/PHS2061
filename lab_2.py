import numpy as np
import monashspa.PHS2061 as spa 
import matplotlib.pyplot as plt

V_s = np.array([0.1 , 0.3 , 0.7 , 1.6 , 3.1 , 5.2 , 7.7 , 10.5 , 13.6 , 16.8 , 20.4 , 24   , 28.2 , 32.1 , 36.2 , 40.4  , 44.8 , 49.2 , 53.8 ,58.4 ,63.0 ,67.5 ,72.2 ,76.8 ,81.5 , 86.5 ,91.3 ,96 ,100.9 ,105.7 ,110.6 ,115.3 ,120.2 ,125 ,129.8 ,134.5 ,139.5 ,154.1, 180.0, 200.0, 220.0, 250.0, 270.0,
    300.0, 330.0, 350.0, 380.0, 410.0, 440.0, 470.0,500.0, 530.0, 560.0, 590.0, 600.0, 610.0, 630.0, 640.0, 660.0,
    690.0, 750.0, 820.0, 850.0, 880.0, 980.0, 1070.0, 1140.0])
V_p = np.array([0.1 , 0.3 , 0.7 , 1.5 , 2.6 , 4.0 , 5.6 , 7.3  , 8.8  , 10.1 , 11.2 , 12.1 , 12.7 , 13.1 , 13.3 , 13.4  , 13.3 , 13.1 , 12.8 , 12.3 , 11.9 ,11.4 ,10.9 ,10.4 ,10.0 ,9.6 ,9.2 ,8.8 ,8.5 ,8.2 ,7.9 ,7.7 ,7.5 ,7.3 ,7.1 , 7.0 ,6.9 ,6.6, 6.3, 6.2, 6.2, 6.3, 6.7,
    7.1, 7.7, 8.5, 9.4, 10.7, 12.6, 15.3,18.2, 21.5, 25.6, 30.9, 34.6, 46.0, 50.7, 116.0, 133.0, 142.0, 180.8,
    250.0, 390.0, 490.0, 550.0, 580.0, 700.0])
v_minus_Vs = np.array([
    100.9e-3, 176.5e-3, 0.301, 0.401, 0.499, 0.597,
    0.696, 0.793, 0.891, 0.985, 1.082, 1.179,
    1.282, 1.377, 1.472, 1.567,1.663 ,1.758 ,1.855 ,1.951 ,2.03 ,2.13 ,2.22 ,2.32 ,2.41 ,2.51 ,2.61 ,2.7 ,2.79 ,2.89 ,2.98 ,3.08 ,3.17 ,3.27 ,3.36 ,3.46 ,3.55 ,3.84, 4.32, 4.79, 5.27, 5.75, 6.22,
    6.7, 7.17, 7.64, 8.11, 8.59, 9.06,
    9.53,10.00, 10.47, 10.94, 11.41, 11.60, 11.79, 11.8, 11.81,
    11.84, 11.91, 12.05, 12.19, 12.25, 12.32, 12.42, 12.43, 12.46
]
)



Vs_0_to_35 =V_s[0:35]
V_p_0_to_35 = V_p[0:35]
v_minus_Vs_0_to_35 = v_minus_Vs[0:35]
print(len(V_s)==len(V_p))
print(len(V_s)==len(v_minus_Vs))

plt.figure(1)
plt.plot(v_minus_Vs,V_s , label = "V-Vs and Vs")

leg = plt.legend(bbox_to_anchor=(1,1))
plt.show()

plt.figure(2)
plt.plot(v_minus_Vs,V_p,label = "V-Vs and Vp")
leg = plt.legend(bbox_to_anchor=(1,1))
plt.show()

plt.figure(3)
plt.plot(v_minus_Vs,V_p,label = "V-Vs and Vp")
plt.plot(v_minus_Vs,V_s , label = "V-Vs and Vs")
leg = plt.legend(bbox_to_anchor=(1,1))
plt.show()

plt.figure(4)
plt.plot(v_minus_Vs_0_to_35,Vs_0_to_35)
plt.plot(v_minus_Vs_0_to_35,V_p_0_to_35)
plt.show()