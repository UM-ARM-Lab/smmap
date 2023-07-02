# smmap
## Dependencies (inherited from smmap_utilities or elsewhere)

* `sudo apt install ros-melodic-moveit-planners ros-melodic-pybind11-catkin`

* [Gurobi](https://www.gurobi.com)  
  Get academic license  
  Download and [follow installation instructions for 9.1.X](http://www.gurobi.com/documentation/9.0/quickstart_linux/software_installation_guid.html#section:Installation) (extract to /opt, add some lines to .bashrc)  
  Switch to using the g++5.2 version 
  
  `cd ${GUROBI_HOME}/lib ` `ln -sf libgurobi_g++5.2.a libgurobi_c++.a`
  
* Python cvxopt

    `sudo apt install python-cvxopt`

* [NOMAD Optimizer](https://www.gerad.ca/software/NOMAD/file?platform=windows&token=63004cb0cc96248ff90d102e262394c3aa8afbb3&version=3.8)

  * Extract files to install directory (typically /opt/nomad.3.8.1)
  * Add `NOMAD_HOME="/opt/nomad.3.8.1"` to `~/.bashrc`
  * run `cd $NOMAD_HOME/install && sudo ./install.sh`
  * Feel free to install somewhere else if you like and change the commands accordingly (https://www.gerad.ca/nomad/Downloads/user_guide.pdf)
