
Exectute these commands in order

cd /home/workspace
cd CarND-Capstone 
pip install -r requirements.txt 
cd ros 
catkin_make 
cd src

cd styx && chmod +x server.py && cd .. && cd styx && chmod +x unity_simulator_launcher.sh && cd .. && cd twist_controller && chmod +x dbw_node.py && cd .. && cd waypoint_loader && chmod +x waypoint_loader.py && cd .. && cd waypoint_updater && chmod +x waypoint_updater.py && cd .. && cd tl_detector && chmod +x tl_detector.py && cd ..

cd ..
source devel/setup.sh 
roslaunch launch/styx.launch





