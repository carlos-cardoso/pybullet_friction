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

import numpy as np
from scipy.spatial.distance import euclidean
import scipy.signal
import theano.tensor as tt
import pymc3 as pm

from fastdtw import fastdtw

print('Running on PyMC3 v{}'.format(pm.__version__))

N_EXPERIMENTS = 1
N_TRIALS = 100
GUI = p.GUI
WATCHDOG = False


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        # sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        # sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def handler(signum, frame):
    print("FROZEN")
    raise ValueError


fname = 'result.bz2'

"""
model	weight(g)	object	description
14	29	lemon	lemon
16	49	pear	pear
17	47	orange	orange
56	58	yball	tennis
57	41	bball	racquet
58	46	wball	golf
65G	28	ocup	medium orange cup
65J	38	ycup	big yellow cup
65D	19	sycup	small yellow cup
73E	26	elego	Lego bridge
73D	16	dlego	Lego eye
"""

models = {
    "lemon": ["14", 29.0],  # lemon
    "pear": ["16", 49.0],  # pear
    "orange": ["17", 47.0],  # orange
    "yball": ["56", 58.0],  # tennis
    "bball": ["57", 41.0],  # racquet
    "wball": ["58", 46.0],  # golf
    "ocup": ["65G", 28.0],  # medium orange cup
    "ycup": ["65J", 38.0],  # big yellow cup
    "sycup": ["65D", 19.0],  # small yellow cup
    "elego": ["73E", 26.0],  # Lego bridge
    "ylego": ["73D", 16.0],  # Lego eye
}


bags = {
    "lemon": list(range(2,32)),
    "spam": list(range(32,62)),
    "ylego": list(range(62,92)),
    "ocup": list(range(94,124)),
}
"""
def load_experiment(fname="effData.txt"):
    colnames = ['toolName', 'targetName', 'actionId', 'initialObjPos[0]', 'initialObjPos[1]', 'initialObjImgPos.x', 'initialObjImgPos.y', 'finalObjectPos[0]',
                'finalObjectPos[1]', 'finalObjImgPos.x', 'finalObjImgPos.y']
    df = pd.read_csv(fname, delim_whitespace=True, names=colnames)

    diffx=(df['finalObjectPos[0]'] - df['initialObjPos[0]'])
    diffy=(df['finalObjectPos[1]'] - df['initialObjPos[1]'])

    mvec = np.array([diffx.mean(), diffy.mean()])
    mdist= np.mean(np.sqrt(diffx**2 + diffy**2))
    vvec = np.array([diffx.var(), diffy.var()])

    weight = 1e-3*models[df['targetName'][0]][1]

    return mvec, vvec, weight, mdist
    """


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

    forces_x = np.array(df_force_field['field.sensorArray6.force.x'][start_ind:end_ind])
    forces_y = np.array(df_force_field['field.sensorArray6.force.y'][start_ind:end_ind])
    forces_z = np.array(df_force_field['field.sensorArray6.force.z'][start_ind:end_ind])

    forces = np.sqrt(forces_x**2 + forces_y**2 + forces_z**2)
    forces = forces-np.mean(forces[0:5])
    #forces = forces_z-np.mean(forces_z[0:5])
    # print(forces_z)

    return object_name[0], object_orientation[0], velocity[0], duration[0], forces


def call_simulator():
    # print("start sim")
    physicsClient = p.connect(GUI)  # or p.DIRECT for non-graphical version

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableFileCaching=0)  # no caching to allow reloading modified urdfs/sdfs
    planeId = p.loadURDF("plane.urdf")


def reset_world():
    p.resetSimulation()


def load_sdf():
    boxId = p.loadSDF("object.sdf")
    return boxId


def load_robot():
    # toolId = p.loadSDF("robot.sdf")
    toolId = p.loadURDF("vizzy.urdf")
    return toolId


def delete_object(objID):
    p.removeBody(objID[0])


def delete_robot(robotID):
    p.removeBody(robotID)


def get_obj_xy(objID):
    pos, ori = p.getBasePositionAndOrientation(objID)
    pos = pos[0:-1]
    return pos


def gen_object(weight=0.3, iner=np.eye(3), center_of_mass=np.zeros((3,)),
               object_name="object", startpose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
    tree = ET.parse('template_object.urdf')
    root = tree.getroot()

    world = root[0]
    # if(world.tag!="world"):
    #    raise Exception("Template world file malformed")
    for mesh in root.findall(".//mesh"):
        mesh.attrib['filename'] = "objects/{}.stl".format(object_name)

    # Set model mass
    for mass in root.findall(".//mass"):
        mass.attrib['value'] = str(weight)
    
    for origin in root.findall(".//origin"):
        origin.attrib["xyz"] = str(center_of_mass)[1:-1]

    for inert in root.findall(".//inertia"):
        inert.attrib['ixx'] = str(iner[0, 0])
        inert.attrib['ixy'] = str(iner[0, 1])
        inert.attrib['ixz'] = str(iner[0, 2])
        inert.attrib['iyy'] = str(iner[1, 1])
        inert.attrib['iyz'] = str(iner[1, 2])
        inert.attrib['izz'] = str(iner[2, 2])

    tree.write('object.urdf')


"""
def gen_robot(action_name, tool_name):
    tree = ET.parse('template_robot_{}.sdf'.format(action_name))
    root = tree.getroot()

    for model in root.findall(".//model[@name='robot']"):
        for uri in model.findall(".//uri"):
            uri.text = "cvx_{}.obj".format(tool_name)
            #print(ET.dump(uri))



    tree.write('robot.sdf')
    """


def cost(pos, des, mdist):
    dist = np.linalg.norm(pos)
    cst = abs(dist - mdist)
    # print("dist: {}, mdist: {}, cost: {}".format(dist, mdist, cst))
    return cst

def find_switch_point(timeseries):
    for ind, val in enumerate(timeseries):
        if val > (timeseries.max()-timeseries.min())/2.0:
            #print("Switchpoint:{}/{}%".format(ind, len(timeseries)))
            return ind
"""

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
            return int(np.mean(trace['tau']))

            lambda_1_samples = trace['lambda_1']
            lambda_2_samples = trace['lambda_2']
            tau_samples = trace['tau']

            #plt.figsize(12.5, 10)
            # histogram of the samples:

            ax = plt.subplot(311)
            ax.set_autoscaley_on(False)

            plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
                     label="posterior of $\lambda_1$", color="#A60628", normed=True)
            plt.legend(loc="upper left")
            plt.title(rPosterior distributions of the variables
                $\lambda_1,\;\lambda_2,\;\tau$)
            plt.xlim([15, 30])
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
"""

def gen_run_experiment(pbar, param_names, object_name="ylego", tools=("rake",), actions=("tap_from_left",)):
    # get properties:
    object_mesh = stl.Mesh.from_file("objects/{}.stl".format(object_name))
    props = object_mesh.get_mass_properties()
    center_of_mass = props[1]
    inertia_tensor = props[2]

    object_bags = bags[object_name]
    gen_object(iner=inertia_tensor,
               center_of_mass=center_of_mass, object_name=object_name, startpose=(0, 0, 0.05, 0, 0, 0))
    # objID = load_sdf()

    objID = p.loadURDF("object.urdf")

    # train_tools = train_tools + test_tools
    # train_actions = train_actions + test_actions

    def f(params):

        dic_params = {pname: params[i] for i, pname in enumerate(param_names)}

        cumulative_cost = 0
        costs = np.zeros((N_EXPERIMENTS,))
        for iter in range(N_EXPERIMENTS):
            for bag in object_bags:
                for action_name in actions:
                    success = False
                    while not success:
                        try:
                            if WATCHDOG:
                                signal.alarm(60)

                            oname, obj_orientation, velocity, duration, forces_z = load_experiment(
                            #    "bags/dsl-dataset-trial_32.bag")
                                "bags/dsl-dataset-trial_{}.bag".format(bag))
                            if(oname != object_name):
                                sys.exit("error: {} !=  {}".format(oname, object_name))

                            # velocity = 4
                            velocity = np.abs(velocity)
                            distance = np.abs(velocity * duration)

                            # redirect stdout required for progress bar
                            with stdout_redirected():
                                toolID = load_robot()

                            # Enable force sensor
                            # p.setJointMotorControl2(bodyUniqueId=toolID,jointIndex=0,controlMode=p.VELOCITY_CONTROL,targetPosition=0,force=1000)
                            p.enableJointForceTorqueSensor(toolID, 0)

                            nonlocal objID
                            if (toolID == objID):
                                raise ValueError
                            # p.resetJointState(toolID[0],0,-0.12,0.0,0)


                            pi = np.pi
                            orientations = {"ylego":[(0.0,0.0,pi), (0.0,0.0,3*pi/2.0)],
                                            "lemon":[(0.0,0.0,-pi/2+pi/4), (0.0,0.0,pi+pi/4)],
                                            "spam":[(0.0,0.0,-0.03+pi/2), (0.0,0.0,-0.03)],
                                            "ocup":[(0.0,0.0,0.0), (0.0,pi/2,0.0)]
                                            }
                            x, y =0.0,0.0
                            yaw, pitch, roll = orientations[object_name][obj_orientation-1]


                            #mu = np.zeros((5,))  # init_poses[tool_name][action_name]
                            #yaw, pitch, roll, x, y = mu  # + np.random.normal(np.zeros_like(mu), np.array([1.0, 1.0, 1.0, 0.01, 0.01]))

                            p.resetBasePositionAndOrientation(objID, posObj=[x, y, 0.05], ornObj=p.getQuaternionFromEuler([yaw, pitch, roll]))
                            dic_params['linearDamping']=0.001
                            dic_params['angularDamping']=0.001
                            p.changeDynamics(toolID, 0, **dic_params)
                            # p.changeDynamics(objID[0], 0, mass=dic_params['mass'])
                            #a = p.getDynamicsInfo(toolID, 0)
                            #print(a)

                            if action_name == "tap_from_left":
                                base_speed = [0, -velocity, 0]
                                base_pos_limit = lambda js: js[1] <= -distance
                            else:
                                raise ValueError

                            # Let object fall to the ground and stop it
                            pxyz, pori = p.getBasePositionAndOrientation(objID)
                            nxyz = 100 * np.ones_like(pxyz)
                            while not np.allclose(nxyz[-1:], pxyz[-1:], atol=1e-6):
                                p.stepSimulation()
                                pxyz = nxyz
                                nxyz, nori = p.getBasePositionAndOrientation(objID)

                            p.resetBasePositionAndOrientation(objID, posObj=pxyz, ornObj=pori)
                            p.resetBaseVelocity(objID, 0)
                            ppos = get_obj_xy(objID)
                            npos = 100 * np.ones_like(ppos)
                            iters = 0

                            # Move tool
                            p.resetBaseVelocity(toolID, base_speed)
                            action_finnished = False
                            simulated_forces = []
                            # while not np.allclose(npos, ppos, atol=1e-6) or iters < 100:
                            while not action_finnished:
                                sensor_reading = p.getJointState(toolID, 0)
                                js, jor = p.getBasePositionAndOrientation(toolID)
                                if (base_pos_limit(js)):
                                    p.resetBaseVelocity(toolID, [0, 0, 0])
                                    action_finnished = True
                                elif not action_finnished:
                                    p.resetBaseVelocity(toolID, base_speed)

                                p.stepSimulation()
                                ppos = npos
                                npos = get_obj_xy(objID)
                                if action_finnished:
                                    iters += 1

                                # contact_points = p.getContactPoints(toolID, objID[0])
                                # sensor_reading = 0
                                # for cp in contact_points:
                                #    sensor_reading +=cp[9]

                                # print(sensor_reading)
                                #simulated_forces.append(sensor_reading[2][1])
                                simulated_forces.append(np.linalg.norm(sensor_reading[2][0:3]))
                                # time.sleep(1./240.)

                            # input("Press Enter to continue...")
                            delete_robot(toolID)


                            simulated_forces =np.array(simulated_forces)
                            simulated_forces = simulated_forces-np.mean(simulated_forces[0:5])

                            switch_point = find_switch_point(forces_z)
                            forces_z = forces_z[switch_point:]

                            switch_point = find_switch_point(simulated_forces)
                            simulated_forces =simulated_forces[switch_point:]

                            simulated_forces = scipy.signal.resample(simulated_forces, len(forces_z))
                            cdistance, path = fastdtw(forces_z[5:-5], simulated_forces[5:-5], dist=euclidean)

                            # print(simulated_forces)
                            #cdistance = np.linalg.norm(simulated_forces[10:-5] - forces_z[10:-5])
                            #print(params)
                            #print(cdistance)
                            # cost(pos - ipos, target_pos, mdist)
                            cumulative_cost += cdistance

                            success = True
                        except ValueError:
                            p.resetSimulation()
                            p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
                            p.setGravity(0, 0, -9.8)
                            p.setPhysicsEngineParameter(enableFileCaching=0)
                            planeId = p.loadURDF("plane.urdf")
                            objID = load_sdf()

            costs[iter] = cumulative_cost
            cumulative_cost = 0

        signal.alarm(0)
        out = np.mean(costs)
        print('\033[93m' + str(dic_params)+'\033[0m')
        # print('\033[92m'+str(out)+'\033[0m')
        pbar.set_description('cost: %0.2f' % (out))
        pbar.update(1)
        return out

    return f


def optimize(param_names, optimizerf):
    # pbounds = {'latf': (0.05, 0.95), 'spif': (0.05, 0.95), 'rollf': (0.05, 0.95), 'rest': (0.05, 0.95), 'weight': (0.010, 0.1)}
    # pbounds = [(0.05, 0.95), (0.05, 0.95), (0.05, 0.95),  (0.05, 0.95), (0.010, 0.1)]

    dbounds = {'lateralFriction': (0.05, 10.0), 'spinningFriction': (0.05, 0.95), 'rollingFriction': (0.05, 0.95),
               'restitution': (0.001, 1.0), 'mass': (0.001, 10.0)}
    # dbounds = {'lateralFriction': (0.05, 100.95), 'spinningFriction': (0.05, 0.95), 'rollingFriction': (0.05, 0.95), 'restitution': (0.05, 0.95), 'mass': (0.010, 100.1)}
    pbounds = [dbounds[param] for param in param_names]

    with tqdm(total=N_TRIALS - 1, file=sys.stdout) as progress_bar:
        run_experiment = gen_run_experiment(progress_bar, param_names)
        res = optimizerf(run_experiment, pbounds, n_calls=N_TRIALS)
        # res = gp_minimize(run_experiment, pbounds, n_calls=N_TRIALS )
        # res = forest_minimize(run_experiment, pbounds, n_calls=100)
        # res = dummy_minimize(run_experiment, pbounds, n_calls=N_TRIALS)
    dump(res, fname, store_objective=False)  # save to file
    return res


def test(param_names):
    with tqdm(total=N_TRIALS - 1, file=sys.stdout) as pbar:
        func = gen_run_experiment(pbar, param_names)

        c_all = []
        costs = []
        best = []
        best_iters = []
        max_target = 1000.0
        res = load(fname)

        for ind, xi in enumerate(res.x_iters):
            pprint(res.func_vals[ind])
            ctarget = res.func_vals[ind]
            c_all.append(ctarget)
            if ctarget < max_target:
                max_target = ctarget
                best.append(ctarget)
                best_iters.append(ind)
                print("new best:{}".format(ctarget))
                params = xi
                pprint(params)
                c = func(params)
                costs.append(c)

    print(c_all)
    print(best)
    print(costs)
    print(ind)


class make_video:
    def __enter__(self):
        out = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "pybullet.mp4")
        return out

    def __exit__(self, type, value, traceback):
        p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, handler)

    call_simulator()

    param_names = ['mass', 'lateralFriction', 'restitution' , 'spinningFriction', 'rollingFriction']

    # forest_minimize
    optimize(param_names, gp_minimize)
    #optimize(param_names, forest_minimize)

    # with make_video:
    #    test(param_names)
    test(param_names)
