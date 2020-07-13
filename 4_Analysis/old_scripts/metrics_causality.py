#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:03:42 2020

@author: li
"""

import pandas as pd
import os
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests






folders = ['1'] # , '2', '3', '4', '5'
faults_type = ['service_cpu', ] # , 'svc_latency', 'service_cpu', 'service_memory'
#    faults_type = ['svc_latency', 'service_cpu']
targets = ['front-end', 'orders', 'carts'  'shipping', 'catalogue', 'user', 'payment']
#targets_svc = ['shipping']



for folder in folders:
    for fault_type in faults_type:
        for target in targets:
            print('=================='+ target + '=====================')
            df = pd.DataFrame()


            if target == 'front-end' and fault_type != 'svc_latency':
                #'skip front-end for service_cpu and service_memory'
                continue

            faults_name = '../2_Copy_Original_data/data/service/' + folder + '/' + fault_type + '_' + target

            latency_filename = faults_name + '_latency_agg_destination_99.csv'
            latency_df_destination = pd.read_csv(latency_filename)


            print(latency_df_destination)

            svc = target

            df['latency'] = latency_df_destination[svc]

            filename = faults_name + '_' + svc + '.csv'
            df_re = pd.read_csv(filename)

            df['cpu'] = df_re['ctn_cpu']
            df['memory'] = df_re['ctn_memory']
            df['network'] = df_re['ctn_network']


            # Granger causality test
            print('Latency & CPU')
            data = df[['latency', 'cpu']].pct_change().dropna()
            gc_res = grangercausalitytests(data, 4)

            print('CPU & Latency')
            data = df[['cpu', 'latency']].pct_change().dropna()
            gc_res = grangercausalitytests(data, 4)

            print('Latency & Memory')
            data = df[['latency', 'memory']].pct_change().dropna()
            gc_res = grangercausalitytests(data, 4)

            print('Memory & Latency')
            data = df[['memory', 'latency']].pct_change().dropna()
            gc_res = grangercausalitytests(data, 4)

            print('Latency & Network')
            data = df[['latency', 'network']].pct_change().dropna()
            gc_res = grangercausalitytests(data, 4)


            print('Network & Latency')
            data = df[['network', 'latency']].pct_change().dropna()
            gc_res = grangercausalitytests(data, 4)



            df['network'].plot()


