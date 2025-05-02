import numpy as np
import monashspa.PHS2061 as spa 
import sympy as sp
import matplotlib.pyplot as plt

V_minus_Vs_to_35 = np.array([100.9e-3,176.5e-3 ,0.301 ,0.401 ,0.499 ,0.597 ,0.696 ,0.793 ,0.891 , 0.985 ,1.082 ,1.179 ,1.282 ,1.377 ,1.472
                       ,1.567 ,1.663 ,1.758 , 1.855 , 1.951 ,2.03 ,2.13 ,2.22 ,2.32 ,2.41 ,2.51 ,2.61 ,2.70 ,2.79 ,2.89 ,2.98 ,3.08 
                       ,3.17 ,3.27 ,3.36 ,3.46 ,3.55 ])
#V
u_V_minus_Vs_to_35 = np.array([0.05,
0.05,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.0005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005
])
#V
Vs_to_35 = np.array([0.1,
0.3,
0.7,
1.6,
3.1,
5.2,
7.7,
10.5,
13.6,
16.8,
20.4,
24,
28.2,
32.1,
36.2,
40.4,
44.8,
49.2,
53.8,
58.4,
63.0,
67.5,
72.2,
76.8,
81.5,
86.5,
91.3,
96,
100.9,
105.7,
110.6,
115.3,
120.2,
125,
129.8,
134.5,
139.5
])
#mV
u_Vs_to_35 = np.array([0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05])
#mV
Vp_to_35 = np.array([0.1,
0.3,
0.7,
1.5,
2.6,
4.0,
5.6,
7.3,
8.8,
10.1,
11.2,
12.1,
12.7,
13.1,
13.3,
13.4,
13.3,
13.1,
12.8,
12.3,
11.9,
11.4,
10.9,
10.4,
10.0,
9.6,
9.2,
8.8,
8.5,
8.2,
7.9,
7.7,
7.5,
7.3,
7.1,
7.0,
6.9
])
#mV
u_Vp_to_35 = np.array([0.05 for i in range(len(Vp_to_35))])
#mV


V_minus_Vs_35_to_125 = np.array([3.84,
4.32,
4.79,
5.27,
5.75,
6.22,
6.7,
7.17,
7.64,
8.11,
8.59,
9.06,
9.53,
10,
10.47,
10.94,
11.41,
11.6,
11.79,
11.8,
11.8,
11.81,
11.81,
11.84,
11.91,
12.05,
12.19,
12.25,
12.32,
12.42,
12.43,
12.46,
12.49])
#V
u_V_minus_Vs_35_to_125 = np.array([0.005 for i in range(len(V_minus_Vs_35_to_125))])
#V
Vs_35_to_125 = np.array([154.1,
0.18e3,
0.20e3,
0.22e3,
0.25e3,
0.27e3,
0.30e3,
0.33e3,
0.35e3,
0.38e3,
0.41e3,
0.44e3,
0.47e3,
0.50e3,
0.53e3,
0.56e3,
0.59e3,
0.6e3,
0.61e3,
0.61e3,
0.63e3,
0.64e3,
0.64e3,
0.66e3,
0.69e3,
0.75e3,
0.82e3,
0.85e3,
0.88e3,
0.98e3,
1.07e3,
1.14e3,
1.21e3
])
#mV
u_Vs_35_to_125 = np.array([0.05,
0.05,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3])
#mV
Vp_35_to_125 = np.array([6.6,
6.3,
6.2,
6.2,
6.3,
6.7,
7.1,
7.7,
8.5,
9.4,
10.7,
12.6,
15.3,
18.2,
21.5,
25.6,
30.9,
34.6,
46.0,
50.7,
116,
133,
142,
180.8,
0.25e3,
0.39e3,
0.49e3,
0.55e3,
0.58e3,
0.70e3,
0.75e3,
0.8e3,
0.86e3])
#mV
u_Vp_35_to_125 = np.array([0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.05,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3])
#mV

V_minus_Vs_total = np.concatenate((V_minus_Vs_to_35,V_minus_Vs_35_to_125))
Vs_total = np.concatenate((Vs_to_35,Vs_35_to_125))
Vp_total = np.concatenate((Vp_to_35,Vp_35_to_125))


V_minus_Vs_in_nitrogen_to_35 = np.array([
100.6e-3,
0.20,
0.3,
0.39,
0.49,
0.59,
0.69,
0.78,
0.88,
0.97,
1.07,
1.16,
1.27,
1.36,
1.45,
1.55,
1.64,
1.74,
1.83,
1.93,
2.03,
2.12,
2.21,
2.31,
2.4,
2.5,
2.59,
2.69,
2.78,
2.88,
2.97,
3.06,
3.16,
3.25,
3.35,
3.44,
3.53])
#V
u_V_minus_Vs_in_nitrogen_to_35 = np.array([0.05e-3,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005,
0.005])
#uV
Vs_in_nitrogen_to_35 = np.array([
0.4e-3,
0.9e-3,
2e-3,
3.7e-3,
6e-3,
8.8e-3,
11.8e-3,
15.1e-3,
18.6e-3,
22.2e-3,
26e-3,
30e-3,
34.6e-3,
38.7e-3,
43.1e-3,
47.6e-3,
52.2e-3,
57e-3,
61.9e-3,
66.9e-3,
72e-3,
77e-3,
82.2e-3,
87.3e-3,
92.5e-3,
98e-3,
103.4e-3,
108.7e-3,
114.2e-3,
119.7e-3,
125.3e-3,
130.8e-3,
136.4e-3,
142e-3,
147.8e-3,
153.4e-3,
159.3e-3
])*1000
#mV
u_Vs_in_nitrogen_to_35 = np.array([0.05 for i in range(len(Vs_in_nitrogen_to_35))])
#mV
Vp_in_nitrogen_to_35 =np.array([
1.2,
2.6,
4.4,
5.6,
6.8,
8.3,
9.9,
11.7,
13.5,
15.4,
17.4,
19.5,
21.8,
24.1,
26.3,
28.7,
31.1,
33.6,
36.1,
38.7,
41.3,
43.8,
46.4,
48.9,
51.5,
54.2,
56.8,
59.4,
62.1,
64.7,
67.4,
70,
72.8,
75.5,
78.3,
81,
83.9])
#mV
u_Vp_in_nitrogen_to_35 = np.array([0.05 for i in range(len(Vp_in_nitrogen_to_35))])
#mV



V_minus_Vs_in_nitrogen_35_to_138 = np.array([
3.82,
4.76,
5.69,
6.62,
7.54,
9.38,
10.3,
11.23,
11.41,
11.69,
13.08])
#V
u_V_minus_Vs_in_nitrogen_35_to_138 = np.array([0.005 for i in range(len(V_minus_Vs_in_nitrogen_35_to_138))])
#V
Vs_in_nitrogen_35_to_138=np.array([0.18,
0.24,
0.3,
0.38,
0.45,
0.62,
0.7,
0.77,
0.79,
0.81,
0.93])*1000
#mV
u_Vs_in_nitrogen_35_to_138 = np.array([0.005e3 for i in range(len(Vs_in_nitrogen_35_to_138))])
#mV
Vp_in_nitrogen_35_to_138 = np.array([
92.6e-3,
122.9e-3,
155e-3,
190.4e-3,
0.22,
0.31,
0.35,
0.39,
0.4,
0.41,
0.51
])*1000
u_Vp_in_nitrogen_35_to_138 = np.array([0.05,
0.05,
0.05,
0.05,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3,
0.005e3])
#mV

V_minus_Vs_in_nitrogen_total = np.concatenate((V_minus_Vs_in_nitrogen_to_35,V_minus_Vs_in_nitrogen_35_to_138))
Vs_in_nitrogen_total = np.concatenate((Vs_in_nitrogen_to_35,Vs_in_nitrogen_35_to_138))
Vp_in_nitrogen_total =np.concatenate((Vp_in_nitrogen_to_35,Vp_in_nitrogen_35_to_138))

def cal_u_for_I(R,uV):
  return(1/R)*uV


Rp = 10000  #ohm
Rs = 100    #ohm

Ip_to_35 = Vp_to_35/Rp #mA
u_Ip_to_35 = cal_u_for_I(Rp,u_Vp_to_35) #mA
Ip_in_nitrogen_to_35 = Vp_in_nitrogen_to_35/Rp  #mA
u_Ip_in_nitrogen_to_35 = cal_u_for_I(Rp,Ip_in_nitrogen_to_35)

Is_to_35 = Vs_to_35/Rs   #mA
u_Is_to_35 =cal_u_for_I(Rs,Is_to_35)#mA
Is_in_nitrogen_to_35 = Vs_in_nitrogen_to_35/Rs  #mA
u_Is_in_nitrogen_to_35 = cal_u_for_I(Rs,Is_in_nitrogen_to_35)#mA

transmission_probability = Ip_to_35/Ip_in_nitrogen_to_35
u_transmission_probability = np.sqrt((u_Ip_to_35/Ip_in_nitrogen_to_35)**2 + (Ip_to_35*u_Ip_in_nitrogen_to_35/(Ip_in_nitrogen_to_35**2))**2)


plt.figure(1)
plt.plot(V_minus_Vs_to_35,Ip_to_35,label = "V-Vs and Ip")
plt.plot(V_minus_Vs_in_nitrogen_to_35,Ip_in_nitrogen_to_35,label = "V-Vs and Ip*")
leg = plt.legend(bbox_to_anchor=(1,1))
plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=12, frameon=True)
plt.xlabel("V-Vs(V)",fontsize=14)
plt.ylabel("I(mA)",fontsize=14)
plt.show()

plt.figure(2)
plt.plot(V_minus_Vs_to_35,transmission_probability,label = "V-Vs and  transmission probability T*")
leg = plt.legend(bbox_to_anchor=(1,1))
plt.xlabel("V-Vs(V)")
plt.ylabel("T")
plt.fill_between(V_minus_Vs_to_35, transmission_probability - u_transmission_probability, transmission_probability + u_transmission_probability, color='blue', alpha=0.3, label='不确定度范围')
plt.show()
print(f"The maximum value of transmission probability is :{np.max(transmission_probability)} | The correspond value for V-Vs is {V_minus_Vs_to_35[np.argmax(transmission_probability)]}")



plt.figure(2)
plt.plot(V_minus_Vs_to_35,Ip_to_35,label = "V-Vs and Ip")

leg = plt.legend(bbox_to_anchor=(1,1))
plt.show()


minus_ln_T = -np.log(transmission_probability)
u_minus_ln_T = np.sqrt((1/transmission_probability)**2 * u_transmission_probability**2)
plt.figure(3)
plt.plot(V_minus_Vs_to_35,minus_ln_T,label = "V-Vs and -ln(T)")
leg = plt.legend(bbox_to_anchor=(1,1))
plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=12, frameon=True)
plt.xlabel("V-Vs(V)",fontsize=14)
plt.ylabel("-ln(T)",fontsize=14)
plt.errorbar(V_minus_Vs_to_35, minus_ln_T, yerr=u_minus_ln_T, fmt='o', label='Data with error bars', color='orange')
plt.show()