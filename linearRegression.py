'''
A python script with the Linear Regression class 
that can be used in the future
'''
import sys
import os
from numpy import array as array

class SimpleLinearRegression:
    #Fields: 
        #x - Independent variable
        #y - Dependent variable
    def __init__(self,x,y):
        self.x = x
        self.y = y
        assert (len(x) != 0 and len(y)!= 0), "x or y cannot be of length 0" 
        assert len(x) == len(y), "x and y are not the same length"
        self.n = len(x)
    def calculate_sum_of_squares(self):
        self.__sumx = sum(x)
        self.__sumy = sum(y)
        self.__sumxsqr = sum([x_i**2 for x_i in x])
        self.__sumysqr = sum([y_i**2 for y_i in y])
        self.__avx = self.__sumx/self.n
        self.__avy = self.__sumy/self.n
        self.__sumxy = sum([x_i*y_i for x_i,y_i in zip(x,y)])
        # sxx and sxy        
        self.sxx = self.__sumxsqr - ((self.__sumx)**2)/self.n
        self.sxy = self.__sumxy - (self.__sumx * self.__sumy)/self.n
        return self.sxx, self.sxy
    def fit(self):
        '''
        Fits the linear model on the data to estimate the beta_0 and beta_1
        '''
        s_xx, s_xy = self.calculate_sum_of_squares()
        self.beta_1 = s_xy/s_xx 
        self.beta_0 = self.__avy - (self.beta_1*self.__avx)
        self.__ssr = self.beta_1*self.sxy 
        self.__sst = self.__sumysqr - self.n * (self.__avy)**2
        self.__ssres = self.__sst - self.__ssr
        self.__msres = self.__ssres/(self.n - 2)
        self.__msr = self.__msres + (self.beta_1**2) * self.sxx
        self.__f0 = self.__msr / self.__msres
    def return_coeff_est(self):
        '''
        Returns the estiate for the regression coefficients
        '''
        assert self.beta_1 , "need to run fit()"
        return [self.beta_1, self.beta_0]

    def predict(self, X = None):
        ''' 
        Outputs a prediction based on the X value given to it
        '''
        if X == None:
            X = self.x
        
        assert self.beta_1 , "need to run fit()"
        return [self.beta_1*x_i + self.beta_1 for x_i in X]

        
    def aov(self):
        assert self.beta_1, "fit() must be run to create ANOVA table" 
        column_headers = ["Sum of Squares", "DOF", "Mean Square", "Fo"]
        row_headers = ["Regression", "Residual", "Total"]
        column_format = "{:>20}" * (len(column_headers) + 1)
        print(column_format.format("", *column_headers))
        #Defining variables
        #Defining data
        data = [[self.__ssr, 1 , self.__msr, self.__f0],[self.__ssres, self.n-2, self.__msres], [self.__sst, self.n -1]]
        rounded_data = [[round(item,2) for item in row] for row in data]
        for row_header, row in zip(row_headers, rounded_data):
            row_format = "{:>20}" * (len(row) + 1)
            print(row_format.format(row_header,*row))

    def __str__(self):
        if self.beta_1:
            return "y ={}*x + {}".format(self.beta_1, self.beta_0) 
        else:
            return "Model has not been fit yet" 

if __name__ == "__main__":
    x = [1,2,3,4,5]
    y = [1,4,9,16,25]
    linearMod = SimpleLinearRegression(x,y)
    linearMod.fit()
    print(linearMod)
    linearMod.aov()
    print(linearMod.predict())