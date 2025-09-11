# online-deflecomp
Online Deflection-Compensation method

## Recommendation
- Ubuntu 20.04 / 22.04
- Python 3.9
- (ROS1)

## Setup (using pyvenv)
```
./setup-environment.sh
./setup-environment-ros.sh
source .venv/bin/activate
```

## Examples

### ROS-Free
```
python3 examples/offline_demo.py 
```

### With ROS1
At terminal 1
```
python ros/sim_node.py --urdf /path/to/simple6r.urdf --dt 0.001 --zeta 3.0 --eq-mode quasistatic --qs-noise-std-deg 0.8 --qs-vib-amp-deg 3.0 --qs-vib-freq-hz 50.0 --qs-vib-axes 1,2 --qs-seed 42
```

At terminal 2
```
python ros/estimator_node.py --urdf /path/to/simple6r.urdf --frames link6
```

At terminal 3 (for visualization)
```
roslaunch ./ros/launch/deflecomp_frames.launch urdf_path:=/path/to/simple6r.urdf
```
