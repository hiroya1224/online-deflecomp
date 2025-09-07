# ROS-lite bundle (no catkin)

This runs from your venv while leveraging system ROS nodes via `roslaunch`.

## Steps

1. Source ROS and activate venv:

```bash
source /opt/ros/noetic/setup.bash
source .venv/bin/activate
pip install online-deflecomp   # robot wrapper, command calc
# pinocchio must be available in this environment
```

2. Start roscore:

```bash
roscore
```

3. Publish prefixed robot_description params (ref_/cmd_/eq_):

```bash
python ros/publish_robot_description.py --urdf /abs/path/to/simple6r.urdf
```

4. Launch GUI + robot_state_publisher x3 + RViz (existing ROS packages only):

```bash
roslaunch ros/launch/deflecomp_rviz_only.launch urdf_path:=/abs/path/to/simple6r.urdf
```

5. Run the simulator from venv (separate terminal):

```bash
./ros/run_sim.sh /abs/path/to/simple6r.urdf
# or:
python ros/sim_node.py --urdf /abs/path/to/simple6r.urdf
```

- /joint_states_ref  : from GUI sliders
- /joint_states_cmd  : GaIK output
- /joint_states_equil: flexible sim result
