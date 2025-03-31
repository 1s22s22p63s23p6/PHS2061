import numpy as np
import sympy as sp
import math 
import monashspa.PHS2061 as spa
import matplotlib.pyplot as plt

class fitting_pool():
    def __init__(self):
        self.seen = []
        self.u_seen = []
        self.deriv_x = None
        self.deriv_y = None
        self.expr_x = None
        self.expr_y = None
        self.symbol = np.arrya(self.seen)
        self.u_symbol = np.array(self.u_seen)
    
    def generate_symbol(self, symbol_dict : dict[str], symbol_dict_for_uncertainty : dict[str]):
        """_summary_

        Args:
            symbol_list (list): _请注意，这个方法的symbol_list中不可以有任何两个相同的符号。
            symbol_list和symbol_list_for_uncertainty中的符号
            这个方法生成用于求不确定度的symbol符号，这些符号将在之后的计算中用来进行求偏微分以及用于x,y的拟合方程_
        """
        
        for sym in symbol_dict:
            if sym in self.seen:
                raise ValueError("There are duplicate elements in symbol_list")
            else:
                self.seen.append[sym]
        
        for u_sym in symbol_dict_for_uncertainty:
            if u_sym in self.u_seen:
                raise ValueError("There are duplicate elements in symbol_list_for_uncertainty")
            else:
                self.u_seen.append[u_sym]
                
        for u_sym in symbol_dict_for_uncertainty:
            if u_sym  not in symbol_dict:
                raise ValueError("The elements in symbol_list and symbol_dict_for_uncertainty are not consistent")
        
        self.__dict__.update({f"{sym}":sp.Symbol(sym) for sym in symbol_dict })
        self.__dict__.update({f"u_{u_sym}":sp.Symbol(u_sym) for u_sym in symbol_dict_for_uncertainty })
        
        
    def generate_expression_and_uncertainty_expression(self,expr_x:str,expr_y:str):
        exprx = sp.sympify(expr_x)
        expry = sp.sympify(expr_y)
        deriv_x = np.array([expr_x.diff(sym) for sym in self.symbol]) 
        deriv_x = (deriv_x * self.u_symbol)
        deriv_x = np.dot(deriv_x,deriv_x)
        deriv_x = np.sqrt(deriv_x)
        deriv_y = np.array([expry.diff(sym) for sym in self.symbol]) 
        deriv_y = (deriv_y * self.u_symbol)
        deriv_y = np.dot(deriv_y,deriv_y)
        deriv_y= np.sqrt(deriv_y)
        x_variable = [sym for sym in self.seen if sym in exprx]
        y_variable = [sym for sym in self.seen if sym in expry]
        expr_for_x = sp.lambdify(x_variable,exprx)
        expr_for_y = sp.lambdify(y_variable,expry)
        self.expr_x()
    equ = "x**2+2*x+1"
    
    derivs = np.array([expr_for_y.diff(sym) for sym in symbols])
derivs = derivs*np.array([s_uI,s_uV,s_ur])
derivs = np.dot(derivs,derivs)
derivs =sp.sqrt(derivs)
print(derivs)