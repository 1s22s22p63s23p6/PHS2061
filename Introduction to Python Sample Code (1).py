# PHS2061 - Introduction to Python Sample Code

# Import relevant functions
import numpy as np
import matplotlib.pyplot as plt
import monashspa.PHS2061 as spa

# Import relevant data, suggested format for import is a .csv 
# (comma-seperated values) file. Note that you man need to use 
# skip_header to remove column headers in the file.
data = np.genfromtxt("fitting_tutorial_data.csv", delimiter=",", skip_header=1)

# You should unpack any imported data, and create any other required values
x_data =
y_data =
u_x_data =
u_y_data =

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
