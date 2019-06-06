# dbounds = {'lateralFriction': (0.05, 0.95), 'spinningFriction': (0.05, 0.95), 'rollingFriction': (0.05, 0.95),
#            'restitution': (0.05, 0.95), 'mass': (0.010, 0.1)}
#dbounds = {'lateralFriction': (0., 100.), 'spinningFriction': (0., 100.), 'rollingFriction': (0., 0.01),
#           'restitution': (0., 1.), 'mass': (0., 0.2)}
dbounds = {'lateralFriction': (0.01, 5.0), 'spinningFriction': (1.0e-4, 5.0), 'rollingFriction': (1.0e-12, 1.0e-3),
             'restitution': (0.0001, 0.95), 'mass': (0.0001, 0.2), 'xnoise' : (0.0, 0.05), 'ynoise' : (0.0, 0.05),
             'xmean' : (-0.02, 0.02), 'ymean' : (-0.02, 0.02)}

# param_names = ['lateralFriction', 'spinningFriction', 'rollingFriction',]# 'restitution']
param_names = ['lateralFriction', 'rollingFriction', 'mass']# 'restitution']

train_tools = ("rake",)
train_actions = ("tap_from_left", "push")
object_name = "ylego"
test_tools = ("hook",)
test_actions = ("tap_from_right",)


N_EXPERIMENTS = 500  # running experiment per object per tool
N_TRIALS = 35  # optimization steps
from skopt import gp_minimize, forest_minimize, dummy_minimize
optimizer = gp_minimize