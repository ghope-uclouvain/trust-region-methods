This is very much a work in progress!

The goal of this project is to make an easy to use and modify Python library for Trust-Region Optimization Methods. The actual trust-region methods are performed in C for speed.

On architecture:
opt_tr is the class that all other trust-region methods inherit from. The idea is that if you want to implement your own, you inherit opt_tr and define your specific methods for updating steps, trust region radius, etc. A template has been included as opt_tr_template if you want the framework to do that.