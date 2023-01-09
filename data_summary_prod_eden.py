########################
# To run locally, execute  `bokeh serve --show data_summary_xxx.py`  (old, prod, dev)
# To run on server, execute `bokeh serve data_summary_prod.py --port 5100  --allow-websocket-origin=fh1-donut01.dun.fh:5100`
#########################

# import numpy as np   # not used
import pandas as pd
import pickle
from math import radians, pi

# Bokeh Library
from bokeh.plotting import figure  # show, output_file, output_notebook, save
from bokeh.models import HoverTool, OpenURL, TapTool, PanTool, ResetTool, WheelZoomTool, Paragraph, SaveTool  # , BoxZoomTool, SaveTool, LassoSelectTool
from bokeh.layouts import column, row  # ,layout
from bokeh.models.widgets import Slider, Select,  RadioButtonGroup,Button, CheckboxGroup, Toggle  # , RangeSlider, CheckboxButtonGroup, TextInput,
from bokeh.io import curdoc
from bokeh.models.sources import ColumnDataSource
from bokeh.models.annotations import Label
import bokeh.palettes as palette

### ### ### filter out hover warnings
import warnings
warnings.filterwarnings("ignore", message="HoverTool are being repeated")

# import master data

master = pickle.load(open("df_output_for_donut_0122.pkl", "rb"))
usage_date = 'Jan 2022'

##################
# set global defaults
##################

width = 850
height = 900

ug_cent = 335
ug_half = 40
ug_over = 5
usage_pal = palette.RdYlGn[4]

show_usage = True

background = 'white'
base_colors = [col for i, col in enumerate(palette.Category20b[20]) if i % 2 == 0]    # takes alternate colors in palette Category20b
rec_alphas = [0.9, 0.6]
color_map = {i: j for i, j in zip(base_colors, [col for i, col in enumerate(palette.Category20b[20]) if i % 2 == 1])}  # maps alternate colors to 'next step' color

# country quick selec t dropdown combinations
country_dd = {'ALL':[i for i in range(18)],
              'UK':[7,9,13,14,15,17],                               #[6,8,12,13,14,16],
              'UK & Ireland':[7,9,10,13,14,15,17],                #[6,8,9,12,13,14,16],
              'Ireland':[10],                                        #[9],
              'Americas':[0,4,12,16],                                      #[0,3,11,15],
              'Australia & NZ':[2,3,11],
              'Asia':[1]
              }


##################################
# Functions used
##################################


def set_default_rads(show_usage=False):
    """
    Defines chart radii for with / without usage variants. returns dict of radii params
    """
    radii = {}
    if show_usage == True:
        radii['radius_0'] = 60
        radii['radius_1'] = 70
        radii['radius_2'] = 150
        radii['radius_3'] = 250
        radii['hint_inc'] = 10
        radii['excl_inc'] = 8
        radii['line_inc_outer'] = 20
    elif show_usage == False:
        radii['radius_0'] = 80
        radii['radius_1'] = 100
        radii['radius_2'] = 200
        radii['radius_3'] = 350
        radii['hint_inc'] = 10
        radii['excl_inc'] = 8
        radii['line_inc_outer'] = 20
    return radii


def angles_from_seg(start, segs):
    """
    to convert segment angles to start/end angles array
    for use *clockwise* from start
    """
    start_angle = pd.Series(start)
    angles = start_angle.append(start_angle[0]-segs.cumsum())  # .map(radians) # from when working in degrees
    return angles


def st_end_angles(angles):
    """
    return start & end angles arrays from 'full' segment angles array
    """
    return angles[:-1], angles[1:]


def as_radians(angles_series):
    """
    return an array of angles in degrees as radians
    """
    return angles_series.map(radians)


def add_sizes(df, col, total=pi*2):
    """
    get size of segments in radians = 'size'
    """
    df['size'] = (df[col] / df[col].sum())*total
    return df


def get_segs(df, start=pi/2):
    """
    returns segment boundary angles (= of # segments +1)
    """
    # print ('get segs', start)
    return angles_from_seg(start, df['size'])


def add_st_end(df, start=pi/2):
    """
    adds start, end angles (in radians) to glyph df
    """
    segs = get_segs(df, start)
    starts, ends = st_end_angles(segs)
    df['start'] = starts.values
    df['end'] = ends.values
    df['mid'] = (starts.values + ends.values)/2  # added for any need to draw centre lines in wedges
    return df


def add_centre_radius(df, inner, outer, hint_inc=0, excl_inc=0):
    """
    add centre & inner, outer to glyph df
    """
    df['centre_x'] = 0
    df['centre_y'] = 0
    df['inner'] = inner
    df['outer'] = outer
    if 'hintable' in df.columns:
        df.loc[df['hintable'] == 'Yes', 'outer'] = outer+hint_inc
    if 'exclusive' in df.columns:
        df.loc[df['exclusive'] == 'Yes', 'inner'] = inner+excl_inc
    return df


def add_color_alphas(df, colors, alphas=[1.0]):
    """
    adds colors & alphas to df
    """
    df['color'] = (colors*df.shape[0])[:df.shape[0]]
    df['alpha'] = (alphas*df.shape[0])[:df.shape[0]]
    return df


def event_string(df, col='events'):
    """
    formats event count as readable friendly string in m
    """
    df['str_from_events'] = ['{:,.1f} m'.format(i/1000000) if (i/1000000) > 10 else
                             '{:,.2f} m'.format(i/1000000) if (i/1000000) > 1 else
                             '{:,.3f} m'.format(i/1000000)for i in df[col]]
    return df


def cat_title(df):
    """
    formats a category title field from index (cat_df)
    """
    df['cat_title'] = df.index.str.title()
    return df


def dataset_title(df):
    """
    formats dataset title field from dataset name
    """
    df['dataset_title'] = df['dataset'].str.title()
    df['dataset_title'] = df['dataset_title'].str.replace('Us', 'US')
    df['dataset_title'] = df['dataset_title'].str.replace('Uk', 'UK')
    df['dataset_title'] = df['dataset_title'].str.replace("'S", "'s")
    return df


def dataset_url(df):
    """
    creates dataset url suffix
    """
    #use dataset_orig to avoid the (dataset_part notation causing issues)
    df['dataset_url'] = df['dataset_orig'].str.replace(',', '')
    df['dataset_url'] = df['dataset_url'].str.replace("'", '')
    df['dataset_url'] = df['dataset_url'].str.replace('(', '')
    df['dataset_url'] = df['dataset_url'].str.replace(')', '')
    df['dataset_url'] = df['dataset_url'].str.replace('&', 'and')
    df['dataset_url'] = df['dataset_url'].str.replace(' ', '-')
    df['dataset_url'] = df['dataset_url'].str.replace('vols\.', 'vols')
    df['dataset_url'] = df['dataset_url'].str.replace('vol\.', 'vol') # added for Gilbert Family History (vol. 6)
    return df


def format_cat_df(df, radii):
    """
    add all additional content & formatting on cat_df
    """
    df = add_sizes(df, 'events')
    df = add_st_end(df)
    df = add_centre_radius(df, radii['radius_1'], radii['radius_2'])
    df = add_color_alphas(df, base_colors)
    df = event_string(df)
    df = cat_title(df)
    # print(df.iloc[0, :])
    return df


def add_usage_rad_col(df):
    """
    add usage data radius & colour
    """
    df['usage_rad'] = ug_cent+(2*(df['ftv_ratio_scaled']-master['ftv_ratio_scaled'].mean())*ug_half) #ftv or totv
    df['usage_rad_totv'] = ug_cent+(2*(df['totv_ratio_scaled']-master['totv_ratio_scaled'].mean())*ug_half) #ftv or totv
    df['usage_col'] = None
    df.loc[df['usage_rad'] > (ug_cent+(ug_half/2)), 'usage_col'] = usage_pal[0]
    df.loc[(df['usage_rad'] <= (ug_cent+(ug_half/2))) &
           (df['usage_rad'] > ug_cent), 'usage_col'] = usage_pal[1]
    df.loc[(df['usage_rad'] <= (ug_cent)) &
           (df['usage_rad'] > ug_cent-(ug_half/2)), 'usage_col'] = usage_pal[2]
    df.loc[(df['usage_rad'] <= (ug_cent-(ug_half/2))), 'usage_col'] = usage_pal[3]
    df['usage_perc'] = ["{:.0f}".format(i) for i in (df['ftv_ratio_scaled']*100)]
    df['usage_perc_totv'] = ["{:.0f}".format(i) for i in (df['totv_ratio_scaled']*100)]
    df['str_from_ftv'] = ['{:,.0f} m'.format(i/1000000) if (i/1000000) > 10 else
                             '{:,.2f} m'.format(i/1000000) if (i/1000000) > 1 else
                             '{:,.3f} m'.format(i/1000000)for i in df['ft_view']]
    df['ftv_index'] = ['{:,.3f}'.format(i*1000) for i in df['ftv_ratio']]
    df['totv_index'] = ['{:,.3f}'.format(i*1000) for i in df['totv_ratio']]
    return df


def format_rec_df(df, cat_total, cat_start, cat_color, radii):
    """
    add all additional content & formatting on a rec_df
    """
    df = add_sizes(df, 'events', total=cat_total)
    df = add_st_end(df, cat_start)
    df = add_centre_radius(df, radii['radius_2'], radii['radius_3'],
                           hint_inc=radii['hint_inc'], excl_inc=radii['excl_inc'])
    colors = color_map[cat_color]
    df = add_color_alphas(df, [colors], alphas=rec_alphas)

    df['cat_title'] = df['category'].str.title()
    return df


def create_rec_df_dict(cat_df, used_data, radii):
    """
    creates dict of formatter recordset CDS (from dfs) - one for each category
    """
    rec_df_dict = {}
    for category in cat_df.index:
        # print(category)
        df = used_data.loc[used_data['category'] == category].copy() #changed to used_data in mast (from master - which shouldn'y have worked, but did ...)
        df.sort_values('events', ascending=False, inplace=True)
        df = format_rec_df(df, cat_df.loc[category, 'size'],
                           cat_df.loc[category, 'start'],
                           cat_df.loc[category, 'color'], radii)
        rec_df_dict[category] = df.to_dict(orient = 'list') #ColumnDataSource.from_df(df)

    return rec_df_dict  # a dict of dicts (CDSs)


def select_data(cat_min, cat_max, rec_min, rec_max, country, hintable=2, exclusive=2,
                recordtype=[0, 1, 2, 3], cat_select='ALL'):
    """
    takes inputs from all widgets, creates masks and selects relevant data from master_df
    """
    # cat size mask
    if cat_select == 'ALL':
        cat_list = (cat_master_df.loc[(cat_master_df['events'] >= cat_min)
                                      & (cat_master_df['events'] <= cat_max)].index)
        cat_mask = (master['category'].isin(cat_list))
    else:
        cat_mask = (master['category'] == cat_select.lower())

    # recordset size mask
    rec_mask = (master['events'] > rec_min) & (master['events'] < rec_max)

    # hintability mask
    if hintable == 2:
        hint_mask = master['hintable'].isin(['Yes', 'No'])
    elif hintable == 1:
        hint_mask = master['hintable'].isin(['No'])
    elif hintable == 0:
        hint_mask = master['hintable'].isin(['Yes'])

    # exclusivity mask
    if exclusive == 2:
        excl_mask = master['exclusive'].isin(['Yes', 'No'])
    elif exclusive == 1:
        excl_mask = master['exclusive'].isin(['No'])
    elif exclusive == 0:
        excl_mask = master['exclusive'].isin(['Yes'])

    # recordtype mask
    type_list = ['Records', 'Documents', 'Articles', 'Images']
    active_types = [type_list[i] for i in recordtype]
    rec_type_mask = master['recordtype'].isin(active_types)

    # country mask
    active_countries = '|'.join([x for i,x in zip(range(len(country_list)), country_list) if i in country])   # removed .lower()
    country_mask = master['source_country_list'].str.contains(active_countries)

    # combining them all
    df = master.loc[cat_mask & rec_mask & hint_mask & excl_mask & rec_type_mask & country_mask]
    return df


def format_cat_and_master(df):
    """
    initial edits & configs on master df and creation of cat_master_df
    """
    #print(df.columns)  # test line
    df.rename(columns={'hintable?': 'hintable'}, inplace=True)
    df_2 = df.groupby('category').agg({'events':'sum'})
    df = event_string(df)
    df = dataset_title(df)
    df = dataset_url(df)
    df['source_country_list'] = df['source_country_list'].str.title()
    df['source_country_list'] = df['source_country_list'].str.replace('Uk', 'UK')
    df['recordtype'] = df['recordtype'].str.title()

    # add usage radius and color data to master df
    df = add_usage_rad_col(df)

    country_list = [x.title().replace('Uk', 'UK')
                    for x in list(set(master['source_country_list'].str.split(', ').sum()))]
    country_list.sort()

    cat_select_menu = [x.title() for x in df_2.index]
    cat_select_menu.insert(0, 'ALL')

    return df, df_2, country_list, cat_select_menu


# format master dataframe and cat_master_df and country_list
master, cat_master_df, country_list, cat_select_menu = format_cat_and_master(master)

# seems to need repeating to work properly ....
warnings.filterwarnings("ignore", message="HoverTool are being repeated")


#################
# Main code
#################

# INPUT widgets
button = Button(label="UPDATE CHART", button_type="success")
output_status = Paragraph()
cat_min = Slider(start=0, end=400, value=0, step=0.5, title="Whole category min items (m)")
cat_max = Slider(start=0.5, end=1100, value=1100, step=0.5, title="Whole category max items (m)")
recordset_min = Slider(start=0, end=2, value=0, step=.001, title="Recordset min items (m)")
recordset_max = Slider(start=0.5, end=280, value=280, step=0.5, title="Recordset max items (m)")   #lifted to 280 for US marriages
recordtype = CheckboxGroup(labels=['Records', 'Documents', 'Articles', 'Images'], active=[0, 1, 2, 3])
hintable = RadioButtonGroup(labels=["Hintable", "Not hintable", "All"], active=2)
hint_tip = Paragraph(text='Wedges extruded to outer circle (line) are hintable',
                     style={'font-size': '80%', 'color': 'grey'})
exclusive = RadioButtonGroup(labels=["Exclusive", "Not exclusive", "All"], active=2)
excl_tip = Paragraph(text='Wedges separated from inner segments are exclusive',
                     style={'font-size': '80%', 'color': 'grey'})

country = CheckboxGroup(labels=country_list, active=[x for x in range(len(country_list))])
country_title = Paragraph(text='Source Country contains:')

cat_select = Select(title="ALL or Single Category view:", value="ALL", options=cat_select_menu)
country_dropdown = Select(title="Country quick selection:", value = "ALL", options=list(country_dd.keys()))

usage_toggle = Toggle(label='SHOW usage', button_type='primary', active=False)

# widget descriptor text outputs # disabled when countries added - no space left....
output_1 = Paragraph()
output_2 = Paragraph()
output_3 = Paragraph()
output_4 = Paragraph()
output_5 = Paragraph()
output_6 = Paragraph()

describe_text = [output_1, output_2, output_3, output_4, output_5, output_6]
# for output in describe_text:
#    output.text = 'Nothing yet ....'

# creates widgets & output column
controls_chg = [cat_min, cat_max,  recordset_min, recordset_max]
controls_click = [recordtype, hintable, hint_tip, exclusive, excl_tip]
controls_click_2 = [Paragraph(), country_dropdown, Paragraph(), country_title, country]  # blank Paragraph is blank line

# NOTE - adding describe_text puts in 'live' counts on inputs for inputs (LH) column
controls = [Paragraph(), cat_select, Paragraph(), usage_toggle, output_status, button]+controls_chg+controls_click  # + describe_text # blank Paragraph is blank line

inputs = column(controls, width=300, height=height)
inputs_2 = column(controls_click_2, width=300, height=height)

# Tooltip & tools
# Tooltips configured as custom html elements

TOOLTIPS_1 = """
    <div>
        <div
        style="width:500px; margin-top: 5px; margin-bottom: 5px"
        </div>
        <div>
            <span style="font-size: 16px; font_weight: bold ">@dataset_title</span>
        </div>
         <div>
            <span style="font-size: 14px;">(@str_from_events items)</span>
        </div>
        <div>
            <span style="font-size: 12px;">Category: @cat_title</span>
        </div>
        <div>
            <span style="font-size: 12px;">Type: @recordtype</span>
        </div>
        <div>
            <span style="font-size: 12px;">Hintable? @hintable    </span>
        </div>
        <div>
            <span style="font-size: 12px;">Exclusive? @exclusive</span>
        </div>

        <br style="margin-bottom:15px;"/>
        <div>
            <span style="font-size: 12px; font_weight: bold ">Source country classifications:</span>
        </div>
        <div>
            <span style="font-size: 11px;">@country_events_contrib</span>
        </div>
        <br style="margin-bottom:15px;"/>
        <div>
            <span style="font-size: 12px; font_weight: bold ">rmid contributions:</span>
        </div>
        <div>
            <span style="font-size: 11px;">@rmid_events_contrib</span>
        </div>
    </div>
"""



TOOLTIPS_2 = """
    <div>
        <div
        style="width:500px; margin-top: 5px; margin-bottom: 5px"
        </div>
        <div>
            <span style="font-size: 16px; font_weight: bold ">@cat_title</span>
        </div>
         <div>
            <span style="font-size: 14px;">(@str_from_events events)</span>
        </div>
    </div>
"""

## This is the one where the usage date (month) needs updating
TOOLTIPS_3 = """
    <div>
        <div
        style="width:500px; margin-top: 5px; margin-bottom: 5px"
        </div>
        <div>
            <span style="font-size: 16px; font_weight: bold ">@dataset_title</span>
        </div>
         <div>
            <span style="font-size: 14px;">(@str_from_events items)</span>
        </div>
        <div>
            <span style="font-size: 12px;">Category: @cat_title</span>
        </div>
        <div>
            <span style="font-size: 12px;">Type: @recordtype</span>
        </div>
        <br style="margin-bottom:15px;"/>
        <div>
            <span style="font-size: 14px; font_weight: bold ">Usage stats (Jan 2022):</span>
        </div>
        <div>
            <span style="font-size: 12px;">Usage percentile (FTV per 1,000 items, in type) : @usage_perc</span>
        </div>
        <div>
            <span style="font-size: 12px;">First time views (FTV): @ft_view{,}</span>
        </div>
        <div>
            <span style="font-size: 12px;">FTV per 1,000 items: @ftv_index</span>
        </div>
        <br style="margin-bottom:5px;"/>
        <div>
            <span style="font-size: 12px;">Usage percentile (TOTV per 1,000 items, in type) : @usage_perc_totv</span>
        </div>
        <div>
            <span style="font-size: 12px;">Total views (TOTV): @tot_view{,}</span>
        </div>
        <div>
            <span style="font-size: 12px;">TOTV per 1,000 items: @totv_index</span>
        </div>
        <br style="margin-bottom:5px;"/>
        <div>
            <span style="font-size: 12px;">FTV / Total View ratio: @ft_totv_ratio</span>
        </div>
        <br style="margin-bottom:15px;"/>
        <div>
            <span style="font-size: 12px; font_weight: bold ">DatasetKey (in fulfillments service) FTV contributions:</span>
        </div>
        <div>
            <span style="font-size: 11px;">@dk_ftv_contrib</span>
        </div>
    </div>
"""

hover = HoverTool(tooltips=TOOLTIPS_1, point_policy='follow_mouse', names=['recordset'])
hover_2 = HoverTool(tooltips=TOOLTIPS_2, point_policy='follow_mouse', names=['cat'])
hover_3 = HoverTool(tooltips=TOOLTIPS_3, point_policy='follow_mouse', names=['usage'])
tools = [hover, hover_2, hover_3, TapTool(),  WheelZoomTool(),  PanTool(), ResetTool(), SaveTool()]
# BoxZoomTool(match_aspect=True), SaveTool() TapTool(),


# taptool URL
url = "https://search.findmypast.co.uk/search-world-Records/@dataset_url"


def plot_chart():
    if usage_toggle.active == True:
        usage_toggle.label = 'HIDE usage'
        radii = set_default_rads(show_usage=True)
    elif usage_toggle.active == False:
        usage_toggle.label = 'SHOW usage'
        radii = set_default_rads(show_usage=False)

    # CREATE FIGURE
    p = figure(plot_width=width, plot_height=height, title="EDEN - findmypast Datasets (at end Jan 2022)",
               x_axis_type=None, y_axis_type=None,
               x_range=(-420, 420), y_range=(-420, 420),
               min_border=0, outline_line_color=None,
               background_fill_color=background,
               tools = tools, toolbar_location="above")

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # collect df from Master df as defined by widget settings
    used_data = select_data(cat_min.value*1000000, cat_max.value*1000000,
                            recordset_min.value*1000000, recordset_max.value*1000000,
                            country.active, hintable.active, exclusive.active,
                            recordtype.active, cat_select.value
                           )

    # in case selections result in a ZERO df
    if used_data.shape[0] == 0:
        # print('NO DATA')
        # prints a '0 items' line
        label = Label(x=-400, y=400, x_offset=0, text='{:,.0f} items'.format(used_data['events'].sum()),
                      text_baseline="middle")
        p.add_layout(label)
        return p

    # format category df, and create recordset dataframes within categories
    cat_df = used_data.groupby('category').agg({'events':'sum'})  # .sort_values('events', inplace=True)
    cat_df.sort_values('events', inplace=True, ascending=False)
    cat_df = format_cat_df(cat_df, radii)
    # format as CDS
    cat_source = ColumnDataSource.from_df(cat_df)

    # create dict of dataframes - 1 for each category containing all the recordsets in that category
    rec_df_dict = create_rec_df_dict(cat_df, used_data, radii)

    # add a circle to highlight with / without hints radius
    p.circle(0,0, radius=radii['radius_3']+radii['hint_inc'], fill_alpha=0, line_color='grey', line_alpha=0.4)

    # plot category  wedges
    cat = p.annular_wedge('centre_x', 'centre_y', 'inner', 'outer', 'start', 'end', color='color',
                          alpha='alpha', direction='clock', source=cat_source, name='cat')

    # Text creation
    # total item count + recordset count + cat count
    item_count = Label(x=-400, y=390, x_offset=0,
                       text='{:,.0f} items'.format(cat_df['events'].sum()),
                       text_baseline="middle", text_font_size = '11pt')

    recordset_count = Label(x=-400, y=372, x_offset=0,
                       text='{:,.0f} datasets'.format(used_data.shape[0]),
                       text_baseline="middle", text_font_size = '11pt')

    cat_count = Label(x=-400, y=354, x_offset=0,
                       text='{:,.0f} categories'.format(cat_df.shape[0]),
                       text_baseline="middle", text_font_size = '11pt')


    counts = [item_count, recordset_count, cat_count]
    for count in counts: p.add_layout(count)


    # Explainers removed into columns (under radiobuttongroup)
    # hint_explain = Label(x=-400, y=-370, x_offset=0, text='Wedges extruded to outer circle (line) are hintable')
    # hint_explain.text_font_size='8pt'
    # p.add_layout(hint_explain)

    # excl_explain = Label(x=-400, y=-384, x_offset=0, text='Wedges separated from inner segments are exclusive')
    # excl_explain.text_font_size = '8pt'
    # p.add_layout(excl_explain)

    # widget setting descriptors
    output_status.text = 'Status: Current'
    output_1.text = 'Minimum items in category: {:,.1f}m'.format(cat_min.value)
    output_2.text = 'Maximum items in category: {:,.1f}m'.format(cat_max.value)
    output_3.text = 'Minimum items in record set: {:,.3f}m'.format(recordset_min.value)
    output_4.text = 'Maximum items in record set: {:,.1f}m'.format(recordset_max.value)
    hintability_list = ["Hintable", "Not hintable", "All"]
    output_5.text = 'Hintability selection: '+ hintability_list[hintable.active]
    rec_type_list = ['Records','Documents','Articles', 'Images']
    active_rec_types = [rec_type_list [i] for i in recordtype.active]
    type_string = ', '.join(active_rec_types)
    output_6.text = 'Active types: '+type_string

    # creates taptool for recordset url links. set renderers to empty list
    taptool = p.select(type=TapTool)[0]
    taptool.renderers = []  # blank out renderers list (from default auto) to only add recordset glpyh sets

    # plot recordset wedges
    for category in cat_df.index:
        records_source = rec_df_dict[category]
        recordset = p.annular_wedge('centre_x', 'centre_y', 'inner', 'outer', 'start', 'end', color='color',
                                    alpha='alpha', direction='clock', source=records_source,
                                    name='recordset', line_width=0)
        taptool.renderers.append(recordset)  # add recordset renderer to renderer list for taptool

    # plot radial category lines
    p.annular_wedge(0, 0, radii['radius_0'], radii['radius_3']+radii['line_inc_outer'],
                    cat_df['start'], cat_df['start'], color="grey")

    # usage grid & bars
    if usage_toggle.active == True:
        p.circle(0,0, radius=ug_cent, fill_alpha=0, line_color='grey', line_alpha=0.9)
        p.circle(0,0, radius=ug_cent+ug_half, fill_alpha=0, line_color='grey', line_alpha=0.4)
        p.circle(0,0, radius=ug_cent-ug_half, fill_alpha=0, line_color='grey', line_alpha=0.4)
        p.circle(0,0, radius=ug_cent-(ug_half/2), fill_alpha=0, line_color='grey', line_alpha=0.2)
        p.circle(0,0, radius=ug_cent+(ug_half/2), fill_alpha=0, line_color='grey', line_alpha=0.2)

        p.annular_wedge(0, 0, ug_cent-ug_half-ug_over, ug_cent+ug_half+ug_over,
                        cat_df['start'], cat_df['start'], color="grey")

        for category in cat_df.index:
            records_source = rec_df_dict[category]
            recordset = p.annular_wedge('centre_x', 'centre_y', ug_cent-ug_half, 'usage_rad', 'start', 'end', color='usage_col',
                                        alpha=0.7, direction='clock', source=records_source,
                                        name='usage', line_width=0.5)
            taptool.renderers.append(recordset)  # add recordset renderer to renderer list for taptool
            # can be used to draw a central line for totv usage percentile
            #p.annular_wedge(0, 0, ug_cent-ug_half, 'usage_rad_totv', 'mid', 'mid', color="grey",
             #                 alpha=0.7, direction='clock', source=records_source, name='usage_mid', line_width=0.2)

    # set taptool callback
    taptool.callback = OpenURL(url=url)

    # return the figure
    return p


def callback(attr, old, new):
    # doesn't change chart - but updates all the widget descriptor text outputs
    output_status.text = 'Status: Changes Pending, press UPDATE'
    # output_1.text = 'Minimum items in category: {:,.1f}m'.format(cat_min.value)
    # output_2.text = 'Maximum items in category: {:,.1f}m'.format(cat_max.value)
    # output_3.text = 'Minimum items in record set: {:,.3f}m'.format(recordset_min.value)
    # output_4.text = 'Maximum items in record set: {:,.1f}m'.format(recordset_max.value)
    # hintability_list = ["Hintable", "Not hintable", "All"]
    # output_5.text = 'Hintability selection: '+ hintability_list[hintable.active]
    # rec_type_list = ['Records', 'Documents', 'Articles', 'Images']
    # active_rec_types = [rec_type_list [i] for i in recordtype.active]
    # type_string = ', '.join(active_rec_types)
    # output_6.text = 'Active types: '+type_string


def callback_2():
    # UPDATER - clears the doc, and calls plot_chart() to create a new one with all the new settings
    curdoc().clear()
    p = plot_chart()
    r = row([inputs_2, inputs, p])
    curdoc().add_root(r)


def callback_3(attr, old, new):
    # for single category selection dropdown - auto-refresh, & sets cat min/max to defaults
    curdoc().clear()
    cat_min.value = 0
    cat_max.value = 1000
    p = plot_chart()
    r = row([inputs_2, inputs, p])
    curdoc().add_root(r)


def callback_4(attr, old, new):
    # for quick country selection dropdown - auto-refresh, & sets country_active to selected combo
    curdoc().clear()
    country.active = country_dd[country_dropdown.value]
    p = plot_chart()
    r = row([inputs_2, inputs, p])
    curdoc().add_root(r)


def callback_5(attr):
    # show / hide usage
    curdoc().clear()
    p = plot_chart()
    r = row([inputs_2, inputs, p])
    curdoc().add_root(r)


# widget change & click detectors
cat_min.on_change('value', callback)
cat_max.on_change('value', callback)
# cat_min_max.on_change('value', callback)
recordset_min.on_change('value', callback)
recordset_max.on_change('value', callback)
hintable.on_change('active', callback)
exclusive.on_change('active', callback)
recordtype.on_change('active', callback)
country.on_change('active', callback)

# category select detector & country quick select dropdown (update immediately)
cat_select.on_change('value', callback_3)
country_dropdown.on_change('value', callback_4)

# update button click detector
button.on_click(callback_2)
usage_toggle.on_click(callback_5)

p = plot_chart()
r = row([inputs_2, inputs, p])


curdoc().add_root(r)
curdoc().title = "findmypast dataset viewer"
