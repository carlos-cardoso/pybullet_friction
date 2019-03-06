#!/usr/bin/python3
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import stl
from pprint import pprint
import pybullet as p
import pybullet_data
import signal
from skopt import gp_minimize, dump, load, forest_minimize, dummy_minimize
from tqdm import tqdm, tqdm_notebook
import os
import sys
from contextlib import contextmanager
import time
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.spatial.distance import euclidean
import scipy.signal
import theano.tensor as tt
import pymc3 as pm

from fastdtw import fastdtw

print('Running on PyMC3 v{}'.format(pm.__version__))


def load_experiment(fname="bags/bag1"):
    # data_info_colnames = ['%time','field.header.seq','field.header.stamp','field.header.frame_id','field.trial_id','field.object_id','field.object_name','field.location_id','field.repetition_num','field.velocity_y','field.movement_duration']
    df_info = pd.read_csv(fname + "_datasetInfo.csv")
    # %time,field.data
    velocity = df_info['field.velocity_y']
    duration = df_info['field.movement_duration']
    object_name = df_info['field.object_name']
    object_orientation = df_info['field.location_id']
    # print(velocity)
    # print(duration)
    # print(object_name)

    df_status = pd.read_csv(fname + "_datasetStatus.csv")

    start_time, end_time = 0, 0
    for ind, action in enumerate(df_status['field.data']):
        if action == 'Performing action':
            start_time = df_status['%time'][ind]
        if action == 'Going to home position':
            end_time = df_status['%time'][ind]

    if start_time == 0 or end_time == 0:
        sys.exit("error in {}, no <Performing action> or <Going to home position>".format(fname))
    # print(start_time)
    # print(end_time)
    # %time,field.header.seq,field.header.stamp,field.header.frame_id,field.sensorArray0.frame_id,field.sensorArray0.force.x,field.sensorArray0.force.y,field.sensorArray0.force.z,field.sensorArray0.displacement.x,field.sensorArray0.displacement.y,field.sensorArray0.displacement.z,field.sensorArray0.field.x,field.sensorArray0.field.y,field.sensorArray0.field.z,field.sensorArray1.frame_id,field.sensorArray1.force.x,field.sensorArray1.force.y,field.sensorArray1.force.z,field.sensorArray1.displacement.x,field.sensorArray1.displacement.y,field.sensorArray1.displacement.z,field.sensorArray1.field.x,field.sensorArray1.field.y,field.sensorArray1.field.z,field.sensorArray2.frame_id,field.sensorArray2.force.x,field.sensorArray2.force.y,field.sensorArray2.force.z,field.sensorArray2.displacement.x,field.sensorArray2.displacement.y,field.sensorArray2.displacement.z,field.sensorArray2.field.x,field.sensorArray2.field.y,field.sensorArray2.field.z,field.sensorArray3.frame_id,field.sensorArray3.force.x,field.sensorArray3.force.y,field.sensorArray3.force.z,field.sensorArray3.displacement.x,field.sensorArray3.displacement.y,field.sensorArray3.displacement.z,field.sensorArray3.field.x,field.sensorArray3.field.y,field.sensorArray3.field.z,field.sensorArray4.frame_id,field.sensorArray4.force.x,field.sensorArray4.force.y,field.sensorArray4.force.z,field.sensorArray4.displacement.x,field.sensorArray4.displacement.y,field.sensorArray4.displacement.z,field.sensorArray4.field.x,field.sensorArray4.field.y,field.sensorArray4.field.z,field.sensorArray5.frame_id,field.sensorArray5.force.x,field.sensorArray5.force.y,field.sensorArray5.force.z,field.sensorArray5.displacement.x,field.sensorArray5.displacement.y,field.sensorArray5.displacement.z,field.sensorArray5.field.x,field.sensorArray5.field.y,field.sensorArray5.field.z,field.sensorArray6.frame_id,field.sensorArray6.force.x,field.sensorArray6.force.y,field.sensorArray6.force.z,field.sensorArray6.displacement.x,field.sensorArray6.displacement.y,field.sensorArray6.displacement.z,field.sensorArray6.field.x,field.sensorArray6.field.y,field.sensorArray6.field.z,field.sensorArray7.frame_id,field.sensorArray7.force.x,field.sensorArray7.force.y,field.sensorArray7.force.z,field.sensorArray7.displacement.x,field.sensorArray7.displacement.y,field.sensorArray7.displacement.z,field.sensorArray7.field.x,field.sensorArray7.field.y,field.sensorArray7.field.z,field.sensorArray8.frame_id,field.sensorArray8.force.x,field.sensorArray8.force.y,field.sensorArray8.force.z,field.sensorArray8.displacement.x,field.sensorArray8.displacement.y,field.sensorArray8.displacement.z,field.sensorArray8.field.x,field.sensorArray8.field.y,field.sensorArray8.field.z,field.sensorArray9.frame_id,field.sensorArray9.force.x,field.sensorArray9.force.y,field.sensorArray9.force.z,field.sensorArray9.displacement.x,field.sensorArray9.displacement.y,field.sensorArray9.displacement.z,field.sensorArray9.field.x,field.sensorArray9.field.y,field.sensorArray9.field.z,field.sensorArray10.frame_id,field.sensorArray10.force.x,field.sensorArray10.force.y,field.sensorArray10.force.z,field.sensorArray10.displacement.x,field.sensorArray10.displacement.y,field.sensorArray10.displacement.z,field.sensorArray10.field.x,field.sensorArray10.field.y,field.sensorArray10.field.z
    df_force_field = pd.read_csv(fname + "_tactileForceField.csv")

    start_ind, end_ind = 0, 0
    for ind, ti in enumerate(df_force_field['%time']):
        if ti > start_time and start_ind == 0:
            start_ind = ind
        if ti > end_time and end_ind == 0:
            end_ind = ind

    #forces_x = np.array(df_force_field['field.sensorArray6.force.x'])
    #forces_y = np.array(df_force_field['field.sensorArray6.force.y'])
    #forces_z = np.array(df_force_field['field.sensorArray6.force.z'])

    #forces_x = forces_x-np.mean(forces_x[0:5])
    #forces_y = forces_y-np.mean(forces_y[0:5])
    #forces_z = forces_z-np.mean(forces_z[0:5])

    forces_x = np.array(df_force_field['field.sensorArray4.force.x'][start_ind:end_ind])
    forces_y = np.array(df_force_field['field.sensorArray4.force.y'][start_ind:end_ind])
    forces_z = np.array(df_force_field['field.sensorArray4.force.z'][start_ind:end_ind])
    #fs = np.array(df_force_field['field.sensorArray4.force.z'][start_ind:end_ind])
    fs = forces_z

    #forces = np.sqrt(forces_x**2 + forces_y**2 + forces_z**2)
    #forces = forces-np.mean(forces[0:5])
    #forces = forces_z-np.mean(forces_z[0:5])
    sns.scatterplot(x=np.arange(len(fs)),y=fs)
    plt.show()
    # print(forces_z)

    return object_name[0], object_orientation[0], velocity[0], duration[0], fs

def find_switch_point(timeseries):
    with pm.Model() as model:

        alpha = 1.0 / timeseries.mean()
        mu1 = pm.Exponential("mu_1", alpha)
        mu2 = pm.Exponential("mu_2", alpha)

        tau = pm.DiscreteUniform("tau", lower=0, upper=len(timeseries) - 1)

        with model:
            idx = np.arange(len(timeseries))  # Index
            mu_ = pm.math.switch(tau > idx, mu1, mu2)
        with model:
            observation = pm.Normal("obs", mu_, observed=timeseries)

        with model:
            #step = pm.Metropolis()
            #trace = pm.sample(10000, tune=5000, step=step)
            trace = pm.sample(10000, njobs=4)
            #return int(np.mean(trace['tau']))

            lambda_1_samples = trace['mu_1']
            lambda_2_samples = trace['mu_2']
            tau_samples = trace['tau']

            #plt.figsize(12.5, 10)
            # histogram of the samples:

            ax = plt.subplot(311)
            ax.set_autoscaley_on(True)

            plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
                     label="posterior of $\lambda_1$", color="#A60628", normed=True)
            plt.legend(loc="upper left")
            plt.title(r"""Posterior distributions of the variables
                $\lambda_1,\;\lambda_2,\;\tau$""")
            #plt.xlim([15, 30])
            plt.xlabel("$\lambda_1$ value")

            ax = plt.subplot(312)
            ax.set_autoscaley_on(True)
            plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
                     label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
            plt.legend(loc="upper left")
            plt.xlabel("$\lambda_2$ value")

            plt.subplot(313)
            w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
            plt.hist(tau_samples, bins=len(timeseries), alpha=1,
                     label=r"posterior of $\tau$",
                     color="#467821", weights=w, rwidth=2.)
            plt.xticks(np.arange(len(timeseries)))

            plt.legend(loc="upper left")
            plt.xlabel(r"$\tau$ (in days)")
            plt.ylabel("probability");
            print(lambda_1_samples)
            print(lambda_2_samples)
            print(tau_samples)
            plt.show()
            print(tau_samples.mean())

bags = {
    "lemon": list(range(2,32)),
    "spam": list(range(32,62)),
    "ylego": list(range(62,92)),
    "ocup": list(range(94,124)),
}

for bag in bags["ocup"]:
    object_name, object_orientation, velocity, duration, forces = load_experiment("bags/dsl-dataset-trial_{}.bag".format(bag))
    find_switch_point(forces)