#!/usr/bin/python3
from __future__ import print_function, division, absolute_import

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import stl
from pprint import pprint
import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
# import signal
from skopt import gp_minimize, dump, load, forest_minimize, dummy_minimize
from tqdm import tqdm, tqdm_notebook
import os
import sys
from contextlib import contextmanager
from matplotlib import pyplot as plt
from joblib import delayed, Parallel
import functools
import multiprocessing

import continuous_kl as kl




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

def with_timeout(timeout):
  def decorator(decorated):
    @functools.wraps(decorated)
    def inner(*args, **kwargs):
      pool = multiprocessing.pool.ThreadPool(1)
      async_result = pool.apply_async(decorated, args, kwargs)
      try:
        return async_result.get(timeout)
      except multiprocessing.TimeoutError:
        print("FROZEN")
        # raise ValueError
        return np.array([np.nan, np.nan])

    return inner

  return decorator


# fname= 'result.bz2'
fname = 'saved/rollf_weight.bz2'

# PHYSICS_ENGINE = "ode"  # "ode" "dart" "bullet"

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


def load_experiment(fname="effData.txt", get_eff_data=False):
  colnames = ['toolName', 'targetName', 'actionId', 'initialObjPos[0]', 'initialObjPos[1]', 'initialObjImgPos.x',
              'initialObjImgPos.y', 'finalObjectPos[0]',
              'finalObjectPos[1]', 'finalObjImgPos.x', 'finalObjImgPos.y']
  df = pd.read_csv(fname, delim_whitespace=True, names=colnames)

  diffx = (df['finalObjectPos[0]'] - df['initialObjPos[0]'])
  diffy = (df['finalObjectPos[1]'] - df['initialObjPos[1]'])

  mvec = np.array([diffx.mean(), diffy.mean()])
  mdist = np.mean(np.sqrt(diffx ** 2 + diffy ** 2))
  vvec = np.array([diffx.var(), diffy.var()])

  weight = 1e-3 * models[df['targetName'][0]][1]

  return (mvec, vvec, weight, mdist, np.vstack([diffx, diffy]).T) if get_eff_data else (mvec, vvec, weight, mdist)


def call_simulator(p):
  # print("start sim")
  # physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version

  p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
  p.setGravity(0, 0, -9.8)
  p.setPhysicsEngineParameter(enableFileCaching=0)
  planeId = p.loadURDF("plane.urdf")


def reset_world(p):
  p.resetSimulation()
  # gazebo_error_handling(["gz", "world", "-r"])

#def load_sdf():
#    #print("load sdf")
#    boxId = p.loadSDF("object.sdf")
#    return boxId
#    #gazebo_error_handling(["gz", "model", "-m", "object", "-f", "object.sdf"])

def load_object(p):
    boxId = p.loadURDF("object.urdf")
    return boxId


def load_robot(p):
  # print("load robot")
  # gazebo_error_handling(["gz", "model", "-m", "robot", "-f", "robot.sdf"])
  toolId = p.loadSDF("robot.sdf")
  return toolId


def delete_object(p, objID):
  # print("delete object")
  # gazebo_error_handling(["gz", "model", "-m", "object", "-d"])
  p.removeBody(objID)


def delete_robot(p, robotID):
  # print("delete robot")
  # gazebo_error_handling(["gz", "model", "-m", "robot", "-d"])
  p.removeBody(robotID[0])


# def move_object(yaw=1.57, pitch=0.0, roll=0.0, x=0.035, y=0.0):
#    print("move object: yaw:{}, x:{}, y:{}".format( yaw, x, y))
#    gazebo_error_handling(["gz", "model", "-m", "object", "-Y", str(yaw), "-P", str(pitch), "-R", str(roll), "-x", str(x), "-y", str(y), "-z", "0.05"])

def get_obj_xy(p, objID):
  # gz model -m object -p
  pos, ori = p.getBasePositionAndOrientation(objID)
  pos = pos[0:-1]
  # print(pos)
  return pos


def gen_object(weight=0.3, mu1=0.3, mu2=0.3, slip=0.0, iner=np.eye(3), center_of_mass=np.zeros((3,)),
               object_name="object", startpose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
  tree = ET.parse('template_object.urdf')
  root = tree.getroot()

  world = root[0]
  # if(world.tag!="world"):
  #    raise Exception("Template world file malformed")
  for mesh in root.findall(".//mesh"):
    mesh.attrib['filename'] = "cvx_{}.stl".format(object_name)

  # Set model mass
  for mass in root.findall(".//mass"):
    mass.attrib['value'] = str(weight)

  for inert in root.findall(".//inertia"):
    inert.attrib['ixx'] = str(iner[0, 0])
    inert.attrib['ixy'] = str(iner[0, 1])
    inert.attrib['ixz'] = str(iner[0, 2])
    inert.attrib['iyy'] = str(iner[1, 1])
    inert.attrib['iyz'] = str(iner[1, 2])
    inert.attrib['izz'] = str(iner[2, 2])

  tree.write('object.urdf')


def gen_robot(action_name, tool_name):
  tree = ET.parse('template_robot_{}.sdf'.format(action_name))
  root = tree.getroot()

  # Set robot pose
  for model in root.findall(".//model[@name='robot']"):
    #    for pose in model.findall(".//pose"):
    #        pose.text = str(sample)[1:-1]
    #        print(ET.dump(pose))
    for uri in model.findall(".//uri"):
      uri.text = "cvx_{}.obj".format(tool_name)
      # print(ET.dump(uri))

  tree.write('robot.sdf')


@with_timeout(6.0)
def single_experiment(dic_params, tool_name, object_name, action_name):
  p = bc.BulletClient(connection_mode=pybullet.DIRECT)
  call_simulator(p)
  objID = load_object(p)

  offset = 0.01 if object_name == "yball" else 0.0

  init_poses = {"rake": {"push": np.array([0.0, 0.0, 0.0, -0.04 - offset, 0.0]),
                         "draw": np.array([0.0, 0.0, 0.0, 0.055 + offset, 0.025]),
                         "tap_from_right": np.array([0.0, 0.0, 0.0, 0.00, -0.145 - offset]),
                         "tap_from_left": np.array([0.0, 0.0, 0.0, 0.00, 0.14 + offset])
                         },
                "stick": {"push": np.array([0.0, 0.0, 0.0, -0.035 - offset, 0.0]),
                          "draw": np.array([0.0, 0.0, 0.0, 0.035 + offset, -0.05]),
                          "tap_from_right": np.array([0.0, 0.0, 0.0, 0.06, -0.065 - offset]),
                          "tap_from_left": np.array([0.0, 0.0, 0.0, 0.06, 0.05 + offset])
                          },
                "hook": {"push": np.array([0.0, 0.0, 0.0, -0.04 - offset, 0.0]),
                         "draw": np.array([0.0, 0.0, 0.0, 0.15 + offset, 0.05]),
                         "tap_from_right": np.array([0.0, 0.0, 0.0, 0.06, -0.075 - offset]),
                         "tap_from_left": np.array([0.0, 0.0, 0.0, 0.09, 0.10 + offset])
                         }

                }
  success = False
  while not success:
    try:
      # signal.alarm(60)


      with stdout_redirected():
        toolID = load_robot(p)


      if (toolID == objID):
        raise ValueError

      mu = init_poses[tool_name][action_name]
      yaw, pitch, roll, x, y = np.random.normal(mu, np.array([1.0, 1.0, 1.0, 0.01, 0.01]))
      ipos = np.array([x, y])

      p.resetBasePositionAndOrientation(objID, posObj=[x, y, 0.05], ornObj=[yaw, pitch, roll, 1])
      p.changeDynamics(objID, 0, **dic_params)
      get_obj_xy(p, objID)

      mu = 0.04
      sigma = 0.001
      speed = np.random.normal(mu, sigma)

      if action_name == "push":
        base_speed = [-speed, 0, 0]
        base_pos_limit = lambda js: js[0] <= -0.12
      elif action_name == "draw":
        base_speed = [speed, 0, 0]
        base_pos_limit = lambda js: js[0] >= 0.12
      elif action_name == "tap_from_left":
        base_speed = [0, speed, 0]
        base_pos_limit = lambda js: js[1] >= 0.12
      elif action_name == "tap_from_right":
        base_speed = [0, -speed, 0]
        base_pos_limit = lambda js: js[1] <= -0.12
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
      ppos = get_obj_xy(p, objID)
      npos = 100 * np.ones_like(ppos)
      iters = 0

      # Move tool
      p.resetBaseVelocity(toolID[0], base_speed)
      action_finnished = False
      while not np.allclose(npos, ppos, atol=1e-6) or iters < 100:
        js, jor = p.getBasePositionAndOrientation(toolID[0])
        if (base_pos_limit(js)):
          p.resetBaseVelocity(toolID[0], [0, 0, 0])
          action_finnished = True
        elif not action_finnished:
          p.resetBaseVelocity(toolID[0], base_speed)

        p.stepSimulation()
        ppos = npos
        npos = get_obj_xy(p, objID)
        if action_finnished:
          iters += 1

      pos = npos

      delete_robot(p, toolID)
      delete_object(p, objID)

      success = True
    except ValueError:
      p.resetSimulation()
      p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
      p.setGravity(0, 0, -9.8)
      p.setPhysicsEngineParameter(enableFileCaching=0)
      planeId = p.loadURDF("plane.urdf")
      objID = load_object(p)

  return pos - ipos


def experiment_setup(params, param_names, pbar, object_name, tools, actions):
  from parametersConfig import N_EXPERIMENTS
  # latf=1.0, spif= 0.01, rollf= 0.01, rest=0.01, weight=0.1
  # latf, spif, rollf, rest, weight = params
  # print(latf, spif, rollf, rest, weight)
  dic_params = {pname: params[i] for i, pname in enumerate(param_names)}


  # actions = ["draw", "tap_from_left", "tap_from_right", "push"]

  # object_mesh = stl.Mesh.from_file("cvx_{}.stl".format(object_name))
  # props = object_mesh.get_mass_properties()
  # center_of_mass = props[1]
  # inertia_tensor = props[2]
  # gen_object(weight=0.1, mu1=0.1, mu2=0.1, slip=0.1, iner=inertia_tensor,
  #            center_of_mass=center_of_mass, object_name=object_name, startpose=(0, 0, 0.05, 0, 0, 0))

  # objID = load_object()

  costs = list()
  sim_eff_history = np.zeros((N_EXPERIMENTS, 2))
  for tool_name in tools:
    for action_name in actions:
      eff_data_path = os.path.join('/media/atabak/MyPassport/website_version_final_result',
                                   tool_name,
                                   object_name,
                                   action_name,
                                   'effData.txt')
      target_pos, target_var, gnd_weight, mdist, real_eff_history = load_experiment(
        eff_data_path, get_eff_data=True)

      gen_robot(action_name, tool_name)

        single_effs = Parallel(n_jobs=2)(delayed(single_experiment)(dic_params,
                                                      tool_name,
                                                      object_name,
                                                      action_name) for i in range(N_EXPERIMENTS))
        # import pdb; pdb.set_trace()
        sim_eff_history = np.array(single_effs, dtype=np.float)
        mask = np.all(np.isnan(sim_eff_history), axis=1)
        sim_eff_history = sim_eff_history[~mask]
      else:
        for iter in range(N_EXPERIMENTS):
          sim_eff_history[iter] = single_experiment(dic_params,
                                                    tool_name,
                                                    object_name,
                                                    action_name) for _ in range(N_EXPERIMENTS))
      # import pdb; pdb.set_trace()
      sim_eff_history = np.array(single_effs, dtype=np.float)
      mask = np.all(np.isnan(sim_eff_history), axis=1)
      sim_eff_history = sim_eff_history[~mask]

      if np.random.rand() < 0.1:
        plt.scatter(real_eff_history[:,1], -real_eff_history[:,0], s=40, c="red", edgecolors='none', label="real")
        plt.scatter(sim_eff_history[:, 1], -sim_eff_history[:, 0], s=40, c="blue", edgecolors='none', label="sim")
        plt.legend(loc=2)
        plt.xlim((-.3, .3))
        plt.ylim((-.3, .3))
        plt.title('action: {}, tool: {}'.format(action_name, tool_name))
        plt.show()
      # import pdb;pdb.set_trace()
      costs.append(kl.KL_from_distributions(real_eff_history, sim_eff_history))

    # costs[iter] = cumulative_cost
    cumulative_cost = 0

  # signal.alarm(0)
  out = sum(costs)/len(costs)
  # print('\033[93m' + str(params)+'\033[0m')
  # print('\033[92m'+str(out)+'\033[0m')
  pbar.set_description('cost: %0.2f' % (out))
  pbar.update(1)
  return out


def gen_run_experiment(pbar,
                       param_names,
                       object_name="yball",
                       tools=("rake", "hook", "stick"),
                       actions=("tap_from_left", "draw", "tap_from_right", "push")):
  # get properties:
  object_mesh = stl.Mesh.from_file("cvx_{}.stl".format(object_name))
  props = object_mesh.get_mass_properties()
  center_of_mass = props[1]
  inertia_tensor = props[2]
  gen_object(weight=0.1, mu1=0.1, mu2=0.1, slip=0.1, iner=inertia_tensor,
             center_of_mass=center_of_mass, object_name=object_name, startpose=(0, 0, 0.05, 0, 0, 0))

  f = functools.partial(experiment_setup,
              param_names=param_names,
              pbar=pbar,
              object_name=object_name,
              tools=tools,
              actions=actions)


  # train_tools = train_tools + test_tools
  # train_actions = train_actions + test_actions

  # def f(mu1 = 0.1, slip=0.0, weight=0.058):


  return f


def optimize(param_names):
  from parametersConfig import dbounds, N_TRIALS

  pbounds = [dbounds[param] for param in param_names]

  with tqdm(total=N_TRIALS - 1, file=sys.stdout) as pbar:
    run_experiment = gen_run_experiment(pbar, param_names)
    res = gp_minimize(run_experiment, pbounds, n_calls=N_TRIALS)
    res.specs['args']['func'] = None #  function can't be saved because it has pbar as input
    # import pdb;pdb.set_trace()
    # res = forest_minimize(run_experiment, pbounds, n_calls=100)
    # res = dummy_minimize(run_experiment, pbounds, n_calls=N_TRIALS)
  dump(res, fname, store_objective=False)
  return res


def test(param_names):
  from parametersConfig import N_TRIALS
  # test_tools = ("hook", "rake")
  test_tools = ("rake",)
  # test_actions = ("draw", "tap_from_left")
  test_actions = ("tap_from_left",)

  with tqdm(total=N_TRIALS - 1, file=sys.stdout) as pbar:
    func = gen_run_experiment(pbar, param_names, tools=test_tools, actions=test_actions)

    c_all = []
    costs = []
    best = []
    best_iters = []
    max_target = 1000.0
    res = load(fname)
    # x [list]: location of the minimum.
    # fun [float]: function value at the minimum.
    # x_iters [list of lists]: location of function evaluation for each iteration.
    # func_vals [array]: function value for each iteration.
    # space [Space]: the optimisation space.
    # specs [dict]: the call specifications.
    # rng [RandomState instance]: State of the random state at the end of minimization.

    for ind, xi in enumerate(res.x_iters):
      # data = json.loads(line)
      pprint(res.func_vals[ind])
      ctarget = res.func_vals[ind]
      c_all.append(ctarget)
      if ctarget < max_target:
        # if ctarget > 0:
        max_target = ctarget
        best.append(ctarget)
        best_iters.append(ind)
        print("new best:{}".format(ctarget))
        params = xi  # data['params']
        pprint(params)
        c = func(params)
        costs.append(c)

  print(c_all)
  print(best)
  print(costs)
  print(ind)


if __name__ == "__main__":
  # signal.signal(signal.SIGALRM, handler)
  from parametersConfig import param_names

  # call_simulator()

  optimize(param_names)
  #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "pybullet.mp4")

  test(param_names)
  #p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
