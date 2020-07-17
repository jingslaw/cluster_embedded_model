# cluster_embedded_model
A tools to build the cluster embedded model for MOLCAS

This is a new version for TOMOLCAS program, which fixed some bugs and give some new physical considerations when constructe the model.
There are 4 scripts and the last one is the main program to build the *cluster embedded model*. The program noticed as 1-3 give some judgement when use choose the parameters in program 4: the radius of cluster and the total sphere should be approperiate, otherwise extremly mistake will be caused in MOLCAS calculation by wrong structure or not correct electric potential on defect ions.

Thanks to Pylada package again, the tools to deal with the strucuture from VASP file are mainly from this package. However, as I tried, Pylada can not be correctly set up in WINDOWS system. So if you want use the program in WINDOWS, just use the tools in method file.
