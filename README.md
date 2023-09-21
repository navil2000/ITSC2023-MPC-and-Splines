# ITSC2023-MPC-and-Splines

This repository houses the essential code components required to implement the method proposed by the author in the article presented at the ITSC2023 conference in Bilbao, titled "Hybrid MPC and Spline-based Controller for Lane Change Maneuvers in Autonomous Vehicles."

On one hand, it provides code for spline generation and its corresponding time parametrization. Traditionally, this task was carried out in C++; however, for this project, it has been migrated to Python, along with the controllers.

On the other hand, you will find the MPC controller class. This coding has been developed by following the steps outlined in the examples of the OSQP framework. I highly recommend reading the framework's documentation for a deeper understanding (https://osqp.org/docs/examples/mpc.html)
