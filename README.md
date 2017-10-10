# smmap
## Dependencies
* [Gurobi](https://www.gurobi.com)  
  Get academic license  
  Download and [follow installation instructions for 7.0.X](http://www.gurobi.com/documentation/7.0/quickstart_linux/software_installation_guid.html#section:Installation) (extract to /opt, add some lines to .bashrc)  
  Switch to using the g++5.2 version 
  
  `cd ${GUROBI_HOME}/lib ` `ln -sf libgurobi_g++5.2.a libgurobi_c++.a`
  
* Python cvxopt

    `sudo apt install python-cvxopt`
