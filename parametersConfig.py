# dbounds = {'lateralFriction': (0.05, 0.95), 'spinningFriction': (0.05, 0.95), 'rollingFriction': (0.05, 0.95),
#            'restitution': (0.05, 0.95), 'mass': (0.010, 0.1)}
dbounds = {'lateralFriction': (0., 100.), 'spinningFriction': (0., 100.), 'rollingFriction': (0., 0.01),
           'restitution': (0., 1.), 'mass': (0., 0.2)}
# param_names = ['lateralFriction', 'spinningFriction', 'rollingFriction',]# 'restitution']
param_names = ['lateralFriction', 'rollingFriction',]# 'restitution']

train_tools = ("rake", "hook", "stick")
train_actions = ("tap_from_left", "tap_from_right", "push")
object_name = "yball"
test_tools = ("rake",)
test_actions = ("tap_from_left",)

N_EXPERIMENTS = 100  # running experiment per object per tool
N_TRIALS = 50  # optimization steps
from skopt import gp_minimize, forest_minimize, dummy_minimize
optimizer = gp_minimize