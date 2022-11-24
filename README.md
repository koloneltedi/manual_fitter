# manual_fitter
Some botched tool to help in fitting some straight slopes on top of heatmaps
As I said... it is botched (and poorly commented and a very ugly use of classes), but I might improve it as time goes on.

To install, open your command line in the directory where you wish to install the package
execute "git clone https://github.com/koloneltedi/manual_fitter.git"
Change directory to the manual_fitter directory
execute "python setup.py develop"

In your python file add the import:
"from manual_fitter import manual_fitter"
once you've loaded the data_set from core tools (e.g.: ds = load_by_uuid(123456)) you can add the following lines:

ds_fitter = manual_fitter.PlotObject(ds,charge_stability=True,differential=False,cubic=True) # Find the parameters in the documentation of the init function 
ds_fitter.make_plot(vmax=None) # Find the parameters in the documentation of this function
