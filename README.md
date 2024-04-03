# data_donut

Code & Data for data donut visualisation - available on 
http://fh1-donut02.dun.fh:5100/data_summary_prod

Visualisation runs from a local data file:
df_output_for_donut_MMYY.pkl
(previous months are in data_archive)

df_output_for_donut files are prepared each month using Jupyter notebook on Jupyter Hub:    
http://fh1-pdp01.dun.fh:8000/user-redirect/lab/tree/shared/cbrake/dataset_summary/dataset_summary.ipynb


### To run on donut server
Runs in conda virtual environment `data_vis`    

Command line to run from Donut server    
`bokeh serve data_summary_prod.py --port 5100  --allow-websocket-origin=fh1-donut02.dun.fh:5100`



