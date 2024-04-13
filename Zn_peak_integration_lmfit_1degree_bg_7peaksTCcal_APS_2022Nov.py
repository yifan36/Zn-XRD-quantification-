# -*- coding: utf-8 -*-
"""

@author: yifan
"""

# For texture analysis of insitu Zn texture depo data 

# 7 peaks are used to calculate TC: 002, 100, 102, 103, 110, 112, 201

# plot 4 peak TC: 002,100,102,103

import numpy as np
import scipy as sp
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import lmfit
import scipy.integrate as spi
import re
import time
from datetime import datetime
import os
import configparser
from tqdm import tqdm
config = configparser.ConfigParser()

#%% lmfit with pseudo_voigt model and 2 degree polynomial background to fit xrd peak

def single_peak_fit(x,y,hint,scan_i): 
    
    def pseudo_voigt(x, amplitude, center, sigma, fraction):
        gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
        lorentzian = amplitude * sigma**2 / ((x - center)**2 + sigma**2)
        return (1 - fraction) * gaussian + fraction * lorentzian
    
    def residual(params, x, y):
        amplitude = params['amplitude'].value
        center = params['center'].value
        sigma = params['sigma'].value
        fraction = params['fraction'].value
        background = params['background_c0'].value + params['background_c1'].value * x  
        return y - (pseudo_voigt(x, amplitude, center, sigma, fraction) + background)
    
    # create a Model for the Pseudo-Voigt profile
    model_pv = lmfit.Model(pseudo_voigt)
    
    # add a polynomial background to the model
    bg = lmfit.models.PolynomialModel(degree=1, prefix='background_')
    model = model_pv + bg
    
    # set initial parameter values and bounds

    if scan_i>=0: 
        params = model.make_params(amplitude=hint['amplitude'], center=hint['center'], sigma=hint['sigma'], fraction=hint['fraction'],
                                       background_c0 = hint['background_c0'], background_c1=hint['background_c1'])
        params['background_c0'].min = -1e6
        params['background_c0'].max = 1e6
        params['background_c1'].min = -1e5
        params['background_c1'].max = 1e5
        params['amplitude'].min = 0
        params['amplitude'].max = 5e5
        params['sigma'].min = 0.012
        params['sigma'].max = 0.02
        params['fraction'].min = 0
        params['fraction'].max = 1
        params['center'].min = hint['center']-0.002
        params['center'].max = hint['center']+0.002
    if scan_i>=1: 
        params['amplitude'].max = hint['amplitude']*2+1

    if scan_i<=3:
        params['center'].min = hint['center']-0.02
        params['center'].max = hint['center']+0.02
        

    else: 
        pass
        # model.set_param_hint('amplitude',value=hint['amplitude'])
        # model.set_param_hint('sigma', value=hint['sigma'])#
        # model.set_param_hint('center',  value = hint['center'])#
        # model.set_param_hint('fraction', value = hint['fraction'], min = 0, max = 1)#
        # model.set_param_hint('background_c0', value = hint['background_c0'])
        # model.set_param_hint('background_c0', value = hint['background_c0'])
        # params = model.make_params() 

    # perform the fit
    
    if scan_i>=0: 
        result = model.fit(y, params, x=x, method='nelder')
        #result = model.fit(y, params, x=x, method='nelder')
        result = model.fit(y, params=result.params, x=x)

    else: 
     #    result = model.fit(y, params, x=x)
        pass
    
    # calculate the integrated intensity

    #peak_area = result.eval_components()['pseudo_voigt'].sum() * (x[1] - x[0])
    # peak_area, _ = spi.quad(pseudo_voigt, x[0], x[-1], args=(result.params['amplitude'].value,
    #                                                              result.params['center'].value,
    #                                                              result.params['sigma'].value,
    #                                                              result.params['fraction'].value))
    
    peak_area= (result.params['amplitude'].value * result.params['sigma'].value * np.sqrt(2*np.pi) * (1-result.params['fraction'].value) + 
                        result.params['amplitude'].value * result.params['sigma'].value * result.params['fraction'].value * 2)

    fitting_params = result.best_values
    
    difference=residual(result.params, x, y)
    
#     new_row = pd.Series(['Peak_Area', integrated_intensity], index=['Parameter', 'Value'])
#     df.loc[len(df)] = new_row

    return fitting_params, peak_area, result.best_fit, difference
    



def double_peak_fit(x, y, hint, scan_i):
    def double_pseudo_voigt(x, amplitude1, center1, sigma1, fraction1, amplitude2, center2, sigma2, fraction2):
        gaussian1 = amplitude1 * np.exp(-(x - center1)**2 / (2 * sigma1**2))
        lorentzian1 = amplitude1 * sigma1**2 / ((x - center1)**2 + sigma1**2)
        gaussian2 = amplitude2 * np.exp(-(x - center2)**2 / (2 * sigma2**2))
        lorentzian2 = amplitude2 * sigma2**2 / ((x - center2)**2 + sigma2**2)
        return (1 - fraction1) * gaussian1 + fraction1 * lorentzian1 + (1 - fraction2) * gaussian2 + fraction2 * lorentzian2
    
    def residual(params, x, y):
        amplitude1 = params['amplitude1'].value
        center1 = params['center1'].value
        sigma1 = params['sigma1'].value
        fraction1 = params['fraction1'].value
        amplitude2 = params['amplitude2'].value
        center2 = params['center2'].value
        sigma2 = params['sigma2'].value
        fraction2 = params['fraction2'].value
        background = params['background_c0'].value + params['background_c1'].value * x
        return y - (double_pseudo_voigt(x, amplitude1, center1, sigma1, fraction1, 
                                        amplitude2, center2, sigma2, fraction2) + background)

    # create a Model for the double Pseudo-Voigt profile
    model_pv = lmfit.Model(double_pseudo_voigt)

    # add a polynomial background to the model
    bg = lmfit.models.PolynomialModel(degree=1, prefix='background_')
    model = model_pv + bg

# set initial parameter values and bounds
    if scan_i >= 0:
        params = model.make_params(amplitude1=hint['amplitude1'], center1=hint['center1'], 
                                   sigma1=hint['sigma1'], fraction1=hint['fraction1'], 
                                   amplitude2=hint['amplitude2'], center2=hint['center2'], 
                                   sigma2=hint['sigma2'], fraction2=hint['fraction2'], 
                                   background_c0=hint['background_c0'], background_c1=hint['background_c1'])
        params['background_c0'].min = -1e6
        params['background_c0'].max = 1e6
        params['background_c1'].min = -1e5
        params['background_c1'].max = 1e5
        params['amplitude1'].min = 0
        params['amplitude1'].max = 5e5
        params['amplitude2'].min = 0
        params['amplitude2'].max = 5e5
        params['sigma1'].min = 0.012
        params['sigma1'].max = 0.02
        params['sigma2'].min = 0.014
        params['sigma2'].max = 0.02
        params['fraction1'].min = 0
        params['fraction1'].max = 1
        params['fraction2'].min = 0
        params['fraction2'].max = 1
        params['center1'].min = hint['center1']-0.002
        params['center1'].max = hint['center1']+0.002
        params['center2'].min = hint['center2']-0.002
        params['center2'].max = hint['center2']+0.002

    if scan_i>=1: 
        params['amplitude1'].max = hint['amplitude1']*2+1
        params['amplitude1'].max = hint['amplitude1']*2+1

    if scan_i<=3:
        params['center1'].min = hint['center1']-0.01
        params['center1'].max = hint['center1']+0.01
        params['center2'].min = hint['center2']-0.01
        params['center2'].max = hint['center2']+0.01

    else:
        pass

    # perform the fit

    if scan_i>=0: 
        result = model.fit(y, params, x=x, method='nelder')
        result = model.fit(y, params=result.params, x=x)

    else: 
        #result = model.fit(y, params, x=x)
        pass 

    # calculate the integrated intensity
    # peak_area1, _ = spi.quad(double_pseudo_voigt, x[0], x[-1], args=(
    #     result.params['amplitude1'].value, result.params['center1'].value, 
    #     result.params['sigma1'].value, result.params['fraction1'].value, 
    #     0,0,1e-8,0))

    # peak_area2, _ = spi.quad(double_pseudo_voigt, x[0], x[-1], args=(0,0,1e-8,0,
    #     result.params['amplitude2'].value, result.params['center2'].value, 
    #     result.params['sigma2'].value, result.params['fraction2'].value))


    peak_area1 = (result.params['amplitude1'].value * result.params['sigma1'].value * np.sqrt(2*np.pi) * (1-result.params['fraction1'].value) + 
                        result.params['amplitude1'].value * result.params['sigma1'].value * result.params['fraction1'].value * 2)


    peak_area2 = (result.params['amplitude2'].value * result.params['sigma2'].value * np.sqrt(2*np.pi) * (1-result.params['fraction2'].value) + 
                        result.params['amplitude2'].value * result.params['sigma2'].value * result.params['fraction2'].value * 2)

    fitting_params = result.best_values
    difference = residual(result.params, x, y)

    return fitting_params, peak_area1,peak_area2, result.best_fit, difference


#%% peak area integration with fixed interval, using lmfit  

def peak_intensities_lmfit(dataframe, intervals,hint):

     image_num=[]
     names=dataframe.columns.values.tolist()
     # print(names)
     print('start extracting image_num:')
     for name in tqdm(names):
          image_num.append(re.sub(r'y_obs_([\S]+?)-(\d{5}).tif_Azm=_0.00',r'\2',name))
     print(image_num)
     image_num.remove('x')
     image_num=[int(x) for x in image_num]
     image_num=pd.DataFrame(image_num,columns=["image_num",])
     
    # get image number from header 
     
    #pd.DataFrame(dataframe).to_csv(filename.split('.')[0]+'_new_x.csv',index=None)
     patterns=list(dataframe.columns.values)
     patterns.pop(0)

     data=dataframe.to_numpy()
    
     x=data[:,0] 
     y=data[:,1:]


     interval_i=0

     df_params_all = pd.DataFrame()
     df_best_fits_all = pd.DataFrame()
     df_differences_all = pd.DataFrame()
     df_integrals_all = pd.DataFrame()

     print('\n')
     print('start fitting XRD curves:')

     for (start, stop) in intervals: 

        xx=x[(start<=x) == (x<=stop)]
        yy=y[(start<=x) == (x<=stop),:]
        
        #area=np.zeros(yy.shape[1])
        
        scan_i=0
        
        df_params = pd.DataFrame(list(hint[interval_i].items()), columns=['Parameter', 'initial_guess'])
        df_best_fits = pd.DataFrame(list(xx), columns=['xx',])
        df_differences = pd.DataFrame(list(xx), columns=['xx',])
        
        integrals=[]

        patterns_seq=list(range(yy.shape[1]))[::-1]

        for i in tqdm(patterns_seq): 

            pattern=patterns[i]
            
            if len(hint[interval_i]) < 7:
                hint[interval_i], peak_area, best_fit, difference =  single_peak_fit(xx,yy[:,i],hint[interval_i],scan_i)
            

                df_param = pd.DataFrame(list(hint[interval_i].items()), columns=['Parameter', pattern])
                
                df_best_fit = pd.DataFrame(list(best_fit), columns=[pattern+'_best_fit'])
                df_difference = pd.DataFrame(list(difference), columns=[pattern+'_difference'])


                df_params=pd.concat([df_params,df_param[pattern]],axis=1)
                df_best_fits=pd.concat([df_best_fits, df_best_fit],axis=1)
                df_differences=pd.concat([df_differences,df_difference],axis=1)

                scan_i += 1

                integrals.append(peak_area)

            
            else:
                hint[interval_i], peak_area1, peak_area2, best_fit, difference =  double_peak_fit(xx,yy[:,i],hint[interval_i],scan_i)

                df_param = pd.DataFrame(list(hint[interval_i].items()), columns=['Parameter', pattern])
                
                df_best_fit = pd.DataFrame(list(best_fit), columns=[pattern+'_best_fit'])
                df_difference = pd.DataFrame(list(difference), columns=[pattern+'_difference'])

               #  print('pattern name: ',pattern, '\t', 'peak_area: ', peak_area)
               #  print('\n')

                df_params=pd.concat([df_params,df_param[pattern]],axis=1)
                df_best_fits=pd.concat([df_best_fits, df_best_fit],axis=1)
                df_differences=pd.concat([df_differences,df_difference],axis=1)

                scan_i += 1

                integrals.append((peak_area1,peak_area2))

        integrals=list(reversed(list(integrals)))
        if len(hint[interval_i]) < 7:
            df_integrals = pd.DataFrame(list(integrals), columns=[str([start,stop])])
            df_integrals_all[str([start,stop])]=df_integrals[str([start,stop])]
        else:
            for j in range(2):
                col=str([start,stop])+f'_{j+1}'
                df_integrals = pd.DataFrame(list(zip(*integrals))[j], columns=[col])
                df_integrals_all[col]=df_integrals[col]

        print('fitted interval: ', str([start,stop]))
        interval_i += 1
        

        df_params_all=pd.concat([df_params_all,df_params],axis=0)
        df_best_fits_all=pd.concat([df_best_fits_all,df_best_fits])
        df_differences_all=pd.concat([df_differences_all,df_differences])   
        # _all is to save data/parameters for each interval 

     df_integrals_all=image_num.join(df_integrals_all)
     

     return df_params_all,df_best_fits_all, df_differences_all, df_integrals_all 


#%%  extract time

def extract_time_abs(file_path,start_depo): 
    
    image=[]
    time_abs=[]
    
    for folder,sub_folders,files in os.walk(file_path): 
        for file in files:
            filename = file_path + '\\' + file
            if file.split('.')[-1]=='metadata':
                config.read(filename)
                image.append(int(config['metadata']['imageNumber']))
                time_abs.append(config['metadata']['timeStamp'])
    time_abs=[float(i) for i in time_abs]
    # Convert datetime string to datetime object
    datetime_object = datetime.strptime(start_depo, '%m/%d/%Y %I:%M:%S %p')

     # Extract absolute time from datetime object
    start_depo_abs = datetime_object.timestamp()

    time_abs=[(i-start_depo_abs+7200)/60 for i in time_abs]
    # +7200 is because the time difference btw chicago/atlanta/san jose

    # time_abs=[(i-start_depo_abs-3600)/60 for i in time_abs]
    # -3600 is because the time difference btw chicago/atlanta
    time_depo= pd.DataFrame(list(zip(image,time_abs)), columns=['image_num', 'time_abs'])
    #depo_time=[i-time_abs(0) for i in time_abs]
    
    return time_depo


# calculate TC of Zn

def calculate_TC(integrals, simulated_Zn):


    # "002 peak","100 peak", "102 peak", "103 peak", 
    # "110 peak","112 peak", "200 peak", "201 peak", "203 peak", 
    # "210 peak", "211 peak" 
    
    average_normalized_Zn_peak_intensity=(integrals["002 peak"]/simulated_Zn["(002)"]+
                                          integrals["100 peak"]/simulated_Zn["(100)"]+
                                          integrals["102 peak"]/simulated_Zn["(102)"]+
                                          integrals["103 peak"]/simulated_Zn["(103)"]+
                                          integrals["110 peak"]/simulated_Zn["(110)"]+
                                          integrals["112 peak"]/simulated_Zn["(112)"]+
                                          integrals["201 peak"]/simulated_Zn["(201)"])*1/7
    
    
    integrals["002 peak_TC"] = (integrals["002 peak"]/simulated_Zn["(002)"])/(average_normalized_Zn_peak_intensity) 
    integrals["100 peak_TC"] = (integrals["100 peak"]/simulated_Zn["(100)"])/(average_normalized_Zn_peak_intensity) 
    integrals["102 peak_TC"] = (integrals["102 peak"]/simulated_Zn["(102)"])/(average_normalized_Zn_peak_intensity) 
    integrals["103 peak_TC"] = (integrals["103 peak"]/simulated_Zn["(103)"])/(average_normalized_Zn_peak_intensity) 
    integrals["110 peak_TC"] = (integrals["110 peak"]/simulated_Zn["(110)"])/(average_normalized_Zn_peak_intensity) 
    integrals["112 peak_TC"] = (integrals["112 peak"]/simulated_Zn["(112)"])/(average_normalized_Zn_peak_intensity) 
    integrals["201 peak_TC"] = (integrals["201 peak"]/simulated_Zn["(201)"])/(average_normalized_Zn_peak_intensity) 

    return integrals



