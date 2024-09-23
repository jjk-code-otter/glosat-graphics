# glosat-graphics

Code used to produce and update a selection of graphics for the GloSAT project.

# Variability schematic

This is a simple graphic that schematically shows the different types of variability that contribute to observed 
variability. Running is simple, assuming you have all the packages installed, type the following at the command line:

`python variability_schematic_tim.py`

# Timeline plotting

This is a tool for plotting labels and images. The code needs a json file and a set of images in the same directory. 
There are examples of the json files in the Timelines/InputImages directory. The json file is a dictionary which 
must contain the following keys: 

* start_year - first year of the timeline
* end_year - last year of the timeline
* labels -  a list of labels. Each label is a three element list with start year, end year and text for the label. The
  start year must be specified, but the end year can be set to "null". The text can be any length and "\n" can be used
  to add a line break if necessary.
* images - a list of image specs. Each spec is a three element list with the year, an offset, and an image filename. The 
  year is the point the image refers to. The offset can be used to draw the image offset along the timeline by a whole
  number of years or set to "null". If all elements are set to "null" then the program will automatically work out where 
  to place each image so that they fit between the start year and end year.

Once that's all set up you can run the timeline plotter like by typing the following at the command line:

`python timeline_plotter.py -c InputImages/example.json -o OutputFigures/time_line.png -r 3`

* -c is the configuration file
* -o is the output file
* -r is the number of rows of images to use. If there are lots of images, they can be spread over any number of rows 
  with images assigned iteratively to the rows. If there are three rows then the images in order will appear on rows 
  1, 2, 3, 1, 2, 3, 1, 2 etc.

Some notes: If there are multiple labels with the same start year then these will be amalgamated into a single label.

# MainGraphics

Set of scripts for producing the graphs that appear in the GloSAT illustrations.

Various input files are needed. Some of these can be downloaded using `get_data.py` but not all of them can be 
downloaded automatically because some websites block automated downloads and some websites insist on using php to 
serve the files so they're not available directly. To run type the following at the command line:

`python get_data.py`

Other files are part of the project or represent large quantities of data that you probably want to manage separately. 
I've followed the GloSAT/HadCRUT directory structure for the obs data underneath a main directory specified by the 
environment variable `$DATADIR`. These include:

* GloSATref.1.0.0.0.analysis.anomalies.ensemble_median.nc - GloSAT analysis ensemble median
* GloSATref.1.0.0.0.noninfilled.anomalies.ensemble_median.nc - GloSAT non-infilled ensemble median
* GloSATref.1.0.0.0.analysis.anomalies.*.nc - the ensemble gridded data
* GloSATref.1.0.0.0.analysis.component_series.global.monthly.nc - the time series from GloSAT
* HadCRUT.5.0.2.0.analysis.summary_series.global.monthly.nc - the time series from HadCRUT5
* HadCRUT.5.0.2.0.analysis.anomalies.*.nc - the ensemble gridded version of HadCRUT
* tas_historical1750_UKESM1-1LL_r*i1p1_175001-201412.nc
* tas_Amon_HadCM3_DataAssimilationMean_r1i1p1_17812008.nc
* tas_Amon_HadCM3_FreeRunning_r*i1p1_178012-200912.nc

Once all the data are in place you can run the individual scripts by typing something like the following at the command 
line:

`python script_to_be_run.py`

The names of the scripts correspond to different elements of the illustration. For example, the scripts that 
generate output for the "coverage" illustration are:

* `Coverage_calculate_coverage.py`
* `Coverage_make_random_grid.py`

and for the NAO_SAM:

* `NAO_SAM_plot_maps.py`
* `NAO_SAM_plot_nao.py`
* `NAO_SAM_plot_sam.py`

You get the idea. There's one small exception to this pattern:

* `Volcanoes_plot_erfs.py`

which plots the volcano ERFs and solar ERF. ERF is Effective Radiative Forcing.

For the most part, these images are intended for inclusion in a larger document so they might not show up well in 
a regular image viewers. Some have a transparent background, some have white text, some black and so on.  