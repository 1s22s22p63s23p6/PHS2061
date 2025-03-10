# PHS2061 - Introduction to Python Sample Code

# Import relevant functions
import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS2061 as spa
import sympy as sp

# Import relevant data, suggested format for import is a .csv 
# (comma-seperated values) file. Note that you man need to use 
# skip_header to remove column headers in the file.
m_v = 7.02 #the voltage for magnitic
u_m_v =0.01 #the uncertainty of the volvage for magnitic
m_current = 1.26 #the current I for magnitic
u_m_current = 0.01 #the uncertainty of current for magnitic
u_magnitic = 0.01 #the uncertainity for 
a_v = np.array([ 299.9 , 280.3 , 260.3, 230.0 , 200.0 , 179.8]) #accelerate voltage
u_a_v = 0.05 #uncertainity of accelerate voltage
r =   np.array([ 7 , 6.5 , 6 , 5 , 4.5 , 3.5])*0.01
u_r = 0.005
mu_0 =  ((4*np.pi)*10**(-7))
N_coil =  130.0
a = 0.15 #radius 
print(r)

s_I = sp.Symbol("I")
s_r = sp.Symbol("r")
s_V = sp.Symbol("V")
s_uI = sp.Symbol("uI")
s_ur = sp.Symbol("ur")
s_uV = sp.Symbol("uV")
symbols = np.array([ s_I ,s_V,s_r]) 

B = ((N_coil*mu_0*s_I)/(((5/4)**(3/2))*a))
#B = (7.8*(10**-4))*s_I
expr_for_x = ((2*s_V)/(B**2))
expr_for_y = s_r**2


derivs = np.array([expr_for_x.diff(sym) for sym in symbols])
derivs = derivs*np.array([s_uI,s_uV,s_ur])
derivs = np.dot(derivs,derivs)
derivs =sp.sqrt(derivs)
print(derivs)

expr_x = sp.lambdify([s_I  ,s_V],expr_for_x)
u_expr_x = sp.lambdify([s_I,s_uI , s_V ,  s_uV],derivs)

derivs = np.array([expr_for_y.diff(sym) for sym in symbols])
derivs = derivs*np.array([s_uI,s_uV,s_ur])
derivs = np.dot(derivs,derivs)
derivs =sp.sqrt(derivs)
print(derivs)

expr_y = sp.lambdify(s_r,expr_for_y)
u_expr_y = sp.lambdify([s_r,s_ur],derivs)

X = expr_x(m_current,a_v)
uX = u_expr_x(m_current, u_m_current , a_v, u_a_v)
Y = expr_y(r)
uY = u_expr_y(r,u_r)


#data = np.genfromtxt("fitting_tutorial_data.csv", delimiter=",", skip_header=1)


# You should unpack any imported data, and create any other required values
x_data = X
y_data = Y
u_x_data = uX
u_y_data = uY

# --------Simple linear fit--------

# Create a model
fit_results = spa.linear_fit(x_data, y_data, u_y=u_y_data)
y_fit = fit_results.best_fit
u_y_fit = fit_results.eval_uncertainty(sigma=1)

# Create a Plot
plt.figure()
plt.title("Figure: Linear fit of raw data")
plt.errorbar(x_data, y_data, xerr=u_x_data, yerr=u_y_data, marker="o", linestyle="None", color="black", label="experiment data")
plt.plot(x_data, y_fit, marker="None", linestyle="-", color="black",label="linear fit")
plt.fill_between(x_data,y_fit-u_y_fit,y_fit+u_y_fit, color="lightgrey",label="uncertainty in linear fit")
plt.xlabel("X")
plt.ylabel("Y")
leg = plt.legend(bbox_to_anchor=(1,1))
#plt.savefig('your_plot_name.png',dpi=600, bbox_extra_artists=(leg,), bbox_inches='tight')
plt.show()

# Print the line of best fit parameters
fit_parameters=spa.get_fit_parameters(fit_results)
print(fit_parameters)
print(f"Slop is {fit_parameters.get("slope")} +- {fit_parameters.get("u_slope")}")
print(f"intercept is {fit_parameters.get("intercept")} +- {fit_parameters.get("u_intercept")}")
'''
# --------Non-linear fit--------

# Establish your model, the model in the activity is provided as an example
nonlinear_model = spa.make_lmfit_model("A_0*exp(-l*x)")
# Establish a guess for the parameters being fitted
nonlinear_params = nonlinear_model.make_params(A_0=**Insert the initial counts here***, l=***Insert the inverse of the time at which the counts have approximatly halved***)
# Creates the model
fit_results = spa.model_fit(nonlinear_model,nonlinear_params,x_data,y_data,u_y=u_y_data)

# Create a plot of the model
plt.figure()
plt.title("Figure: Non-linear fit of raw data")
plt.errorbar(x_data, y_data, xerr=u_x_data, yerr=u_y_data, marker="o", linestyle="None", color="black", label="raw data")
plt.plot(x_data, fit_results.best_fit, marker="None", linestyle="-", color="black",label="nonlinear fit")
plt.fill_between(x_data,fit_results.best_fit-fit_results.eval_uncertainty(sigma=1),fit_results.best_fit+fit_results.eval_uncertainty(sigma=1), color="lightgrey",label="uncertainty in nonlinear fit")
plt.xlabel("X")
plt.ylabel("Y")
leg = plt.legend(bbox_to_anchor=(1,1))
#plt.savefig('your_plot_name.png',dpi=600, bbox_extra_artists=(leg,), bbox_inches='tight')
plt.show()

# Print the line of best fit parameters
fit_parameters=spa.get_fit_parameters(fit_results)
print(fit_parameters)
'''