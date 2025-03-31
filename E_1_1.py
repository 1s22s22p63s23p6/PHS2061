# PHS2061 - Introduction to Python Sample Code

# Import relevant functions
import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS2061 as spa
import sympy as sp

# Import relevant data, suggested format for import is a .csv 
# (comma-seperated values) file. Note that you man need to use 
# skip_header to remove column headers in the file.

m_current = 1.305 #the current I for magnitic
u_m_current = 0.05 #the uncertainty of current for magnitic

a_v = np.array([299.9, 280.4,260.3,230,200,179.8]) #accelerate voltage
u_a_v = 0.05 #uncertainity of accelerate voltage
r =   np.array([ 7 , 6.5 , 6 , 5 , 4.5 , 3.5])*0.01
u_r = np.array([0.25,0.3,0.25,0.2,0.16,0.12]) * 0.01
mu_0 =  ((4*np.pi)*10**(-7))
N_coil =  130.0
a = 0.15 #radius 

s_I = sp.Symbol("I")
s_r = sp.Symbol("r")
s_V = sp.Symbol("V")
s_uI = sp.Symbol("uI")
s_ur = sp.Symbol("ur")
s_uV = sp.Symbol("uV")
symbols = np.array([ s_I ,s_V,s_r]) 

B = ((N_coil*mu_0*s_I)/(((5/4)**(3/2))*a))
#B = (7.8*(10**-4))*s_I
expr_for_x = 1/(2*s_V)
expr_for_y = 1/((s_r**2)*(B**2))


derivs = np.array([expr_for_x.diff(sym) for sym in symbols])
derivs = derivs*np.array([s_uI,s_uV,s_ur])
derivs = np.dot(derivs,derivs)
derivs =sp.sqrt(derivs)
print(derivs)

expr_x = sp.lambdify([s_V ],expr_for_x)
u_expr_x = sp.lambdify([s_V ,  s_uV],derivs)

derivs = np.array([expr_for_y.diff(sym) for sym in symbols])
derivs = derivs*np.array([s_uI,s_uV,s_ur])
derivs = np.dot(derivs,derivs)
derivs =sp.sqrt(derivs)
print(derivs)

expr_y = sp.lambdify([s_r,s_I],expr_for_y)
u_expr_y = sp.lambdify([s_r,s_ur,s_I,s_uI],derivs)

X = expr_x(a_v)
uX = u_expr_x( a_v, u_a_v)
Y = expr_y(r,m_current)
uY = u_expr_y(r,u_r,m_current,u_m_current)


#data = np.genfromtxt("fitting_tutorial_data.csv", delimiter=",", skip_header=1)


# You should unpack any imported data, and create any other required values
x_data = X
y_data = Y
u_x_data = uX
u_y_data = uY


# --------Simple linear fit--------

# Create a model
fit_results = spa.linear_fit(x_data, y_data,u_y=u_y_data)
y_fit = fit_results.best_fit
u_y_fit = fit_results.eval_uncertainty(sigma=1)

# Create a Plot
plt.figure()
plt.title("Figure: Linear fit of raw data")
plt.errorbar(x_data, y_data, xerr=u_x_data, yerr=u_y_data, marker="o", linestyle="None", color="black", label="experiment data")
plt.plot(x_data, y_fit, marker="None", linestyle="-", color="black",label="linear fit")
plt.fill_between(x_data,y_fit-u_y_fit,y_fit+u_y_fit, color="lightgrey",label="uncertainty in linear fit")
plt.xlabel("1/2v")
plt.ylabel("1/((r^2)(b^2))")
leg = plt.legend(bbox_to_anchor=(1,1))
#plt.savefig('your_plot_name.png',dpi=600, bbox_extra_artists=(leg,), bbox_inches='tight')
plt.show()

# Print the line of best fit parameters
fit_parameters=spa.get_fit_parameters(fit_results)
print(fit_parameters)
print(f"e/m {fit_parameters.get("slope"):.2f} +- {fit_parameters.get("u_slope"):.2f}")
print(f"intercept is {fit_parameters.get("intercept"):.2f} +- {fit_parameters.get("u_intercept"):.2f}")
print((fit_parameters.get("slope"))/(1.76*(10**11)))


B = ((N_coil*mu_0*m_current)/(((5/4)**(3/2))*a))
#B = (7.8*(10**-4))*s_I
expr_for_x_1 = 1/(2*a_v)
expr_for_y_1 = 1/((r**2)*(B**2))

fit_result_1 = spa.linear_fit(x_data, y_data, u_y=0.1)
fit_parameters_1 = spa.get_fit_parameters(fit_result_1)
print(f"e/m {fit_parameters_1.get("slope"):.2f} +- {fit_parameters_1.get("u_slope"):.2f}")

print(X == expr_for_x_1)

