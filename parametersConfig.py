# dbounds = {'lateralFriction': (0.05, 0.95), 'spinningFriction': (0.05, 0.95), 'rollingFriction': (0.05, 0.95),
#            'restitution': (0.05, 0.95), 'mass': (0.010, 0.1)}
dbounds = {'lateralFriction': (0., 1.), 'spinningFriction': (0., 1.), 'rollingFriction': (0., 1.),
           'restitution': (0., 1.), 'mass': (0., 0.2)}
param_names = ['mass', 'lateralFriction']  # , 'spinningFriction', 'rollingFriction', 'restitution']

N_EXPERIMENTS = 20  # running experiment per object per tool
N_TRIALS = 50  # optimization steps