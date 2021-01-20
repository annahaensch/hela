import altair as alt
import pandas as pd

alt.data_transformers.disable_max_rows()

TU_COLORS = [
    "#9b67ff", "#57b3c2", "#ffa040", "#ff6283", "#2ccc72", "#1270cb", "#bd043a",
    "#a1c54d", "#4c319e", "#c6c0fe", "#195036", "#f6a39f"
]

alt.renderers.enable('notebook')


def draw_states(data, finite_observations=None, hide_brush=False):
    """Shaded bar chart of operating states over time.

    This plot draws a single line for every timestamp. Each color represents
    a new hidden state.

    Args:
        data (pd.Series or pd.Dataframe): dataframe to draw. If data is a series w
        only timestamps and finite_observations, no other arg needed. If more columns present,
        finite_observations should be specified by column header.
        finite_observations (str): name of column containing finite_observations (if dataframe)
        hide_brush (bool): if False, show a brush-to-zoom control at bottom

    Returns:
        Plot of hidden states over time.
    """

    if isinstance(data, pd.Series):
        data = data.to_frame()

    # format the df for Altair and identify column w labels
    df_formatted = data.reset_index()
    states_for_plot = df_formatted.columns[1]
    df_for_plot = df_formatted.rename(columns={
        'index': 'time',
        states_for_plot: 'hidden state'
    })
    states_for_plot = df_for_plot.columns[1]

    if finite_observations is not None:
        states_for_plot = finite_observations

    height = 150

    if not hide_brush:

        brush = alt.selection(type='interval', encodings=['x'])

        chart = alt.Chart(
            df_for_plot, width=900, height=height).mark_tick(
                size=height * 2, opacity=1, thickness=2).encode(
                    alt.X(
                        'time:T',
                        axis=alt.Axis(title=None),
                        scale=alt.Scale(domain=brush),
                    ),
                    y=alt.value(0),
                    tooltip=['time:T', states_for_plot + ':N'],
                    color=alt.Color(
                        states_for_plot + ':N',
                        legend=alt.Legend(
                            orient='bottom', title='hidden states'),
                        scale=alt.Scale(scheme='spectral')))

        brush_chart = alt.Chart(
            df_for_plot, width=900, height=height / 6).mark_tick(
                size=height / 1.5, opacity=0, thickness=2).encode(
                    alt.X(
                        'time:T',
                        axis=alt.Axis(title=None),
                        scale=alt.Scale(domain=brush),
                    ),
                    y=alt.value(0)).add_selection(brush)

        return chart & brush_chart

    else:
        chart = alt.Chart(
            df_for_plot, width=900, height=height).mark_tick(
                size=height / 2, opacity=1, thickness=2).encode(
                    alt.X('time:T', axis=alt.Axis(title=None)),
                    y=alt.value(0),
                    tooltip=['time:T', states_for_plot + ':N'],
                    color=alt.Color(
                        states_for_plot + ':N',
                        legend=alt.Legend(
                            orient='bottom', title='hidden states'),
                        scale=alt.Scale(scheme='spectral')))

        return chart


def draw_raw_features(dataframe,
                      continuous_observations,
                      finite_observations=None,
                      height=50,
                      width=700,
                      line_opacity=0.9,
                      color_scheme=None):
    """Creates stacked line charts with an interactive brush to zoom filter.

    Requires dataframe and an array of column names to plot (no more than 5 at
    a time).  Operating modes appear as a shaded bar near the brush filter. You
    can add more than one categorical features by passin an arr to `finite_observations`.

    Args:
        df: a dataframe to draw
        continuous_observations (arr): array of colum names (str) to include in the chart.
        No more than 5 permitted for a single plot.
        finite_observations (str or arr): the column(s) to use as colors to shade different states.
        Defaultes to None, which only plots line charts and a brush-to-zoom.
        height (int): height of each individual subplot
        width (int): width of entire chart
        line_opacity (float): opacity of color for the main measurement line
        color_scheme (dict): custom color mapping to match colors with labels

    Returns:
        A subplot containing continuous_observations using a shared x-axis, where the
        bottom shading chart acts as a brush to zoom and filter the line
        charts.
    """

    data = dataframe.reset_index()
    df = data.rename(columns={'index': 'time'})

    # check if custom color scheme dict
    if color_scheme is None:
        if finite_observations is not None:
            if isinstance(finite_observations,
                          list):  # take first item if array
                domains = domains = df[finite_observations[0]].unique().tolist()
            else:
                domains = df[finite_observations].unique().tolist()
                ranges = TU_COLORS  # default tagup colors

    else:
        domains = list(color_scheme.keys())
        ranges = list(color_scheme.values())

    brush = alt.selection(type='interval', encodings=['x'])

    ## loop through continuous vars and store line charts
    chartList = []
    d = {}
    for i in continuous_observations:

        line_chart = alt.Chart(df).mark_line(
            color='#143766', size=1.5, opacity=line_opacity).encode(
                alt.X(
                    'time:T',
                    scale=alt.Scale(domain=brush),
                    title=None,
                    axis=alt.Axis(labelFontSize=10, grid=False)),
                y=alt.Y(
                    i + ':Q',
                    axis=alt.Axis(
                        grid=False,
                        titlePadding=10,
                        titleAngle=0,
                        titleY=-5,
                        titleX=8,
                        title=i),
                ),
                tooltip=['time:T', i + ':Q']).properties(
                    height=height, width=width)

        d["{0}".format(i)] = line_chart
        chartList.append(d[i])

    ## if multiple finite observations, do the same for those
    if isinstance(finite_observations, list):
        modeList = []
        color_schemes = ['set1', 'set2', 'set3']

        for f in range(len(finite_observations)):
            selected_ob = finite_observations[f]

            mode_df = pd.concat(
                [df['time'], df[selected_ob]], axis=1, sort=False)

            modes = alt.Chart(mode_df).mark_tick(
                size=height * 2, opacity=1, width=50, thickness=2).encode(
                    alt.X(
                        'time:T',
                        scale=alt.Scale(domain=brush),
                        axis=alt.Axis(title=selected_ob, labelFontSize=10)),
                    y=alt.value(0),
                    tooltip=['time:T', selected_ob],
                    color=alt.Color(
                        selected_ob + ':N',
                        legend=alt.Legend(orient='right', title=selected_ob),
                        scale=alt.Scale(scheme=color_schemes[f]))).properties(
                            width=width, height=height)
            modeList.append(modes)
        modes = modeList[0]

        # take all finite observation charts and stack them in modes object
        for y in range(len(modeList)):
            if y > 0:
                modes = modes & modeList[y]

        # make brush for all the modes and line charts
        shade_brush = alt.Chart(
            df, height=20).mark_tick(
                size=20, opacity=0.5).encode(
                    alt.X(
                        'time:T',
                        axis=alt.Axis(
                            title='brush-to-zoom', titleFontWeight=600)),
                    y=alt.value(8),
                    color=finite_observations[0] + ':N',
                ).properties(
                    width=width,
                    height=20,
                ).add_selection(brush)

    # below fires if finite_observations is str (single column)
    elif isinstance(finite_observations, str):
        mode_df = pd.concat(
            [df['time'], df[finite_observations]], axis=1, sort=False)

        modes = alt.Chart(mode_df).mark_tick(
            size=height * 2, opacity=1, width=50, thickness=2).encode(
                alt.X(
                    'time:T',
                    scale=alt.Scale(domain=brush),
                    axis=alt.Axis(title=None)),
                y=alt.value(0),
                tooltip=['time:T', finite_observations],
                color=alt.Color(
                    finite_observations + ':N',
                    legend=alt.Legend(orient='right', title='states'),
                    scale=alt.Scale(domain=domains, range=ranges))).properties(
                        width=width, height=height)

        shade_brush = alt.Chart(
            df, height=20).mark_tick(
                size=20, opacity=0.5).encode(
                    alt.X(
                        'time:T',
                        axis=alt.Axis(
                            title='brush-to-zoom', titleFontWeight=600)),
                    y=alt.value(8),
                    color=finite_observations + ':N',
                ).properties(
                    width=width,
                    height=20,
                ).add_selection(brush)

    elif finite_observations is None:
        shade_brush = alt.Chart(
            df, height=20).mark_tick(
                size=20, opacity=0.1).encode(
                    alt.X(
                        'time:T',
                        axis=alt.Axis(
                            title='brush-to-zoom', titleFontWeight=600)),
                    y=alt.value(8)).properties(
                        width=width,
                        height=20,
                    ).add_selection(brush)

    # loop through line charts and stack them
    chart = d[continuous_observations[0]]
    for i in range(len(chartList)):
        if i >= 6:
            raise Exception("Too many plots! Choose <= 5 measurements.")
        if i > 0:
            chart = chart & d[continuous_observations[i]]

    if finite_observations is not None:
        return chart & modes & shade_brush
    else:
        return chart & shade_brush


def draw_hidden_state_probs(df):
    """Plot showing behavior of the diagonals in a matrix

    For each hidden state, this plot shows the movement of the diagonal entries
    of the transition matrix over training iterations.

    Args:
        df (pd.Dataframe): dataframe to draw, where each column is a a state

    Returns:
        Step plot of transition matrix.
    """

    df_formatted = df.reset_index()
    melted = pd.melt(df_formatted, id_vars="index")

    chart = alt.Chart(
        melted, width=800).mark_line(
            point=True, interpolate='cardinal', size=2, opacity=0.3).encode(
                x=alt.X('index:O', axis=alt.Axis(title='iteration')),
                y=alt.Y(
                    'value:Q',
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(title='probability')),
                color=alt.Color(
                    'variable', legend=alt.Legend(orient='top',
                                                  title='states')))

    return chart


def draw_validation_metrics(accuracy_categorical,
                            avg_log_likelihood,
                            avg_z_score,
                            best_accuracy_categorical=2,
                            best_avg_log_likelihood=0,
                            best_avg_z_score=0,
                            bounds_accuracy_categorical=0,
                            bounds_avg_log_likelihood=-10,
                            bounds_avg_z_score=3.3):
    """Plot of validation metrics for HMM imputation.

    This plot shows the resulting imputation validation for three metrics. The
    results are plotted in relation to the 'best' possible outcome for each.

    Args:
        accuracy_categorical (int): calculation of the relative accuracy of
        imputed categorical data
        avg_log_likelihood (int): calculation of average relative log likelihood
        of imputed gaussian data
        avg_z_score (int): calculation of average z score of actual data
        best_accuracy_categorical (int): best possible score
        best_avg_log_likelihood (int): best possible score
        best_avg_z_score (int): best possible score
        bounds_accuracy_categorical (int): outer bound of acceptability
        bounds_avg_log_likelihood (int): outer bound of acceptability
        bounds_avg_z_score (int): outer bound of acceptability

    Returns:
        Dot plot showing imputation validation.
    """

    d = {
        'metric': [
            'relative accuracy of imputed categorical data',
            'avg relative log likelihood of imputed gaussian data',
            'avg z score of actual'
        ],
        'result': [accuracy_categorical, avg_log_likelihood, avg_z_score],
        'best':
        [best_accuracy_categorical, best_avg_log_likelihood, best_avg_z_score],
        'outer bounds': [
            bounds_accuracy_categorical, bounds_avg_log_likelihood,
            bounds_avg_z_score
        ]
    }
    validations = pd.DataFrame(data=d)

    domain_ = (bounds_avg_log_likelihood - 2, bounds_avg_z_score + 2)

    circles = alt.Chart(
        validations, height=200, width=800).mark_circle(
            size=85, color='#FF7F50', opacity=1).encode(
                x='result:Q', y='metric:N')

    ticks = alt.Chart(validations).mark_tick(
        size=30, color='#1070CA', thickness=3).encode(
            x=alt.X(
                'best:Q',
                scale=alt.Scale(domain=domain_),
                axis=alt.Axis(grid=False, title=None)),
            y='metric:N')

    bands = alt.Chart(
        validations, height=200, width=800).mark_tick(
            size=30, color='gray', opacity=1, thickness=3).encode(
                x=alt.X(
                    'outer bounds:Q',
                    scale=alt.Scale(),
                    axis=alt.Axis(grid=False, title=None)),
                y='metric:N')

    lines = alt.Chart(validations).mark_rule(
        size=5, color='gray', opacity=0.2).encode(
            x='outer bounds:Q', x2='best:Q', y='metric:N')

    melted = pd.melt(validations, id_vars="metric")
    filt = melted['variable'] == 'best'

    best_label = alt.Chart(melted[filt].iloc[[1]]).mark_text(
        size=10, dy=-22, dx=10, color='#1070CA').encode(
            x=alt.X(
                'value:Q',
                scale=alt.Scale(domain=domain_),
                axis=alt.Axis(grid=False)),
            y='metric:N',
            text=alt.Text('variable:N'))

    filt = melted['variable'] == 'result'

    circ_label = alt.Chart(melted[filt].iloc[[1]]).mark_text(
        size=10, dy=-15, dx=10, color='#FF7F50').encode(
            x=alt.X(
                'value:Q',
                scale=alt.Scale(domain=domain_),
                axis=alt.Axis(grid=False)),
            y='metric:N',
            text=alt.Text('variable:N'))

    bounds_label = alt.Chart(
        melted[melted['variable'] == 'outer bounds'].iloc[[1]]).mark_text(
            size=10, dy=-22, dx=30, color='black').encode(
                x=alt.X(
                    'value:Q',
                    scale=alt.Scale(domain=domain_),
                    axis=alt.Axis(grid=False)),
                y='metric:N',
                text=alt.Text('variable:N'))

    chart = bands + lines + circles + ticks + best_label + circ_label + bounds_label
    return chart.configure_axis(labelLimit=1000)


def draw_imputed_validation(data, data_imputed, redacted_index):
    """Validation plots of imputed values compared to the actual value

    This function calls a plot for each column of data in a DataFrame. Circles
    are drawn for actual and imputed values, represented by diff colors. An
    interpolated line is also added to plots with quantitative data to show
    the natural trend of variation (using a basis interpolation). Columns with
    categorical variabels are represented using circles and ticks to avoid overlap.

    Args:
        data (pd.DataFrame): the full, original dataframe of values
        data_imputed (pd.DataFrame): df with the missing values imputed based on
         model parameters and the redacted_index
        redacted_index (pd.DatetimeIndex): index of rows that should be redacted

    Returns:
        Scatter plot series with interpolated line
    """

    # compare impute and actual values where they are redacted
    # can take either str or int, no need to specify
    width = 600

    # format df for Altair and add type of value
    imputed_melted = pd.melt(
        data_imputed.loc[redacted_index].reset_index(),
        id_vars='index',
        var_name='variable')
    imputed_melted['type'] = 'imputed'

    # format df and isolate only redacted rows
    redacted_melted = pd.melt(
        data.loc[redacted_index].reset_index(),
        id_vars='index',
        var_name='variable')
    redacted_melted['type'] = 'actual'

    # concat these dfs
    df_for_plot = pd.concat([imputed_melted, redacted_melted])

    # isolate unique vars
    cols = df_for_plot.variable.unique()

    chartList = []
    d = {}
    for col in cols:

        filt = df_for_plot.variable == col

        # if categorical data, change plot shapes to avoid overlap
        if isinstance(df_for_plot[filt].value.iloc[1], str):
            data_type = 'N'

            sel = df_for_plot[df_for_plot['variable'] == col]
            piv = pd.pivot_table(
                sel,
                values='variable',
                index=['value'],
                columns=['type'],
                aggfunc='count')
            piv.reset_index(inplace=True)
            piv = piv.rename(columns={'value': 'category'})
            # this is what you need to plot as bar chart
            piv_melt = pd.melt(piv, id_vars=['category'])

            stacked_bars = alt.Chart(
                piv_melt, width=width, height=25).mark_bar(
                    opacity=0.9,).encode(
                        x='value:Q',
                        y=alt.Y(
                            'type:' + data_type,
                            axis=alt.Axis(title=None, grid=False)),
                        color='type:N',
                        row=alt.Row(
                            'category:N',
                            header=alt.Header(labelOrient='top'))).properties(
                                title=col)

            combo_chart = stacked_bars

        else:
            data_type = 'Q'

            circles = alt.Chart(
                df_for_plot, width=width, height=250).mark_circle(
                    size=25, opacity=0.6).encode(
                        x=alt.X('index:T', axis=alt.Axis(title=None)),
                        y=alt.Y(
                            'value:' + data_type,
                            scale=alt.Scale(zero=False),
                            axis=alt.Axis(title=None)),
                        color=alt.Color(
                            'type:N', legend=alt.Legend(orient='right'))
                    ).transform_filter(alt.datum.variable == col)

            lines = alt.Chart(df_for_plot).mark_line(
                size=2, opacity=0.8, interpolate='basis').encode(
                    x='index:T',
                    y=alt.Y(
                        'value:' + data_type,
                        scale=alt.Scale(zero=False),
                        axis=alt.Axis(title=None)),
                    color=alt.Color('type:N')).transform_filter(
                        alt.datum.variable == col).properties(title=col)

            interpolation_label = alt.Chart(df_for_plot.iloc[[1]]).mark_text(
                size=12, dy=-15, dx=0, color='#848383').encode(
                    y=alt.value(0),
                    x=alt.value((width / 2) + 150),
                    text=alt.value('~ interpolation lines'))

            act_label = alt.Chart(df_for_plot.iloc[[1]]).mark_text(
                size=12, dy=-15, dx=31, color='#57B4C3').encode(
                    y=alt.value(0),
                    x=alt.value(width / 2),
                    text=alt.value('actual'))

            imp_label = alt.Chart(df_for_plot.iloc[[1]]).mark_text(
                size=12, dy=-15, dx=-20, color='#FF9F40').encode(
                    y=alt.value(0),
                    x=alt.value(width / 2),
                    text=alt.value('imputed'))

            combo_chart = circles + lines + interpolation_label + act_label + imp_label

        d["{0}".format(col)] = combo_chart
        chartList.append(d[col])

    chart = d[cols[0]]

    for i in range(len(chartList)):

        if i > 0:
            chart = chart & d[cols[i]]

    return chart


# NOTE:
# - the tooltip feature for plot belot currently only works with Altair 3.2.0.
# When upgrading to Altair 4.0, the tooltip does not appear for multiple subplots.
# I think this is an issue on the Vega-Lite side.
# - tooltips do not show when displaying this type of chart in Streamlit
def draw_status_curve_chart(
        df,
        states,  # categorical variable for shading
        curves,  # arr of columns to plot as curves (must plot at least one)
        conditioning_date,  # string that is a datestamp
        color_mapping='default',  # pass a dict with keys as states, values as hex codes
        highlighted_states=None,  # if None, all states after conditioning date will be semi-transparent. to highlight states, pass an arr of strings here.
        save_as_html=False):
    """Shaded bar plot overlaid with curves

    This function creates a plot with a few features: the main plot contains shaded
    bars for every timestamp given a certain "state" or category. The "curves" arg
    will plot one (or multiple) curves on top of this shaded bar plot. Below the plot
     is a brush for zooming into the plot at different ranges.

    Args:
        df (pd.DataFrame): a dataframe which is indexed by timestamps and contains
         1) categorical data to plot as shaded bars and 2) at least one column with
         values to plot as a line overlaid ontop of the plot
        states (str): title of column in df with categorical data
        curves: (arr): list of column names as strings to plot as curves. If only one,
         enter the arg as an arr with one string
        conditioning_date (str): datestamp of conditioning
        color_mapping (dict): option to create a custom color scheme for shaded bars.
        If passing a dict, keys should match the unique strings contained in "states"
        column while values should be HEX codes or color names.
        highlighted_states (arr): option to add an arr of column names to "highlight",
        which will increase the opacity of these states any time it appears after
        the conditioning date
        save_as_html (Bool): if True, will save a standalone HTML file of the chart
        with interactivity and data included.

    Returns:
        Shaded bar plot overlaid with curves
    """

    chart_df = df.reset_index()

    ### ASSIGN A COLOR MAP ###

    COLORS = [
        "#57b4c3", "#004284", "#262262", "#0364ff", "#2ecc71", "#9966ff",
        "yellow"
    ]

    colors_for_curves = ['#FF7F50', 'black', 'gray', 'pink']

    # these will be passed to the chart
    if color_mapping == 'default':
        domains = list(chart_df[states].unique())
        ranges = COLORS
    else:
        domains = list(color_mapping.keys())
        ranges = list(color_mapping.values())

    ### SET CONDITIONAL OPACITY ###

    opacity_values = []
    default_opacity = 0.7

    # loop through and set an opacity value for each row
    for index, row in chart_df.iterrows():
        # for all rows before the conditioning date, set a default opacity value
        if row['index'] < pd.to_datetime(conditioning_date):
            opacity_values.append(default_opacity)
        # if highlighed states is specified, make these states more opaque
        elif highlighted_states is not None:
            if row[states] in highlighted_states:
                opacity_values.append(1)
            else:
                opacity_values.append(0.2)
        # if no specified highlighted states, use default low opacity for everything after the conditioning date
        elif highlighted_states is None:
            opacity_values.append(0.2)

    chart_df['highlight'] = opacity_values

    ### ALTAIR CODE TO CREATE CHARTS ###

    top_chart_height = 100
    chart_width = 700

    brush = alt.selection(type='interval', encodings=['x'])

    def draw_curve(df, col, line_color):
        curve = alt.Chart(
            df, width=chart_width).mark_line(
                color=line_color, opacity=1, size=2, clip=True).encode(
                    x=alt.X('index:T', scale=alt.Scale(domain=brush)),
                    y=alt.Y(
                        col + ':Q',
                        axis=alt.Axis(title=None),
                        scale=alt.Scale(domain=(0, df[col].max()))))

        return curve

    if len(curves) == 1:
        tooltips = [
            alt.Tooltip(
                curves[0] + ':Q',
                format='.2'),  # if %, add a % sign after the 2 in format
            alt.Tooltip(states + ':N', format='')
        ]
    elif len(curves) == 2:
        tooltips = [
            alt.Tooltip(curves[0] + ':Q', format='.2'),
            alt.Tooltip(curves[1] + ':Q', format='.2'),
            alt.Tooltip(states + ':N', format='')
        ]
    elif len(curves) == 3:
        tooltips = [
            alt.Tooltip(curves[0] + ':Q', format='.2'),
            alt.Tooltip(curves[1] + ':Q', format='.2'),
            alt.Tooltip(curves[2] + ':Q', format='.2'),
            alt.Tooltip(states + ':N', format='')
        ]
    elif len(curves) == 4:
        tooltips = [
            alt.Tooltip(curves[0] + ':Q', format='.2'),
            alt.Tooltip(curves[1] + ':Q', format='.2'),
            alt.Tooltip(curves[2] + ':Q', format='.2'),
            alt.Tooltip(curves[3] + ':Q', format='.2'),
            alt.Tooltip(states + ':N', format='')
        ]

    # chart with shaded cateogories is the backdrop of everything
    chart = alt.Chart(
        chart_df, height=top_chart_height, width=chart_width).mark_tick(
            thickness=1, size=140, opacity=0.8).encode(
                x=alt.X('index:T', scale=alt.Scale(domain=brush)),
                y=alt.value((top_chart_height / 2) + 12),
                color=alt.Color(
                    states + ':N',
                    legend=alt.Legend(orient='right', labelLimit=200),
                    scale=alt.Scale(domain=domains, range=ranges)),
                opacity=alt.Opacity(
                    'highlight:Q', legend=None, scale=alt.Scale(range=(0.1,
                                                                       1))),
                tooltip=tooltips)

    if len(curves) == 1:
        curve_plots = draw_curve(
            chart_df, curves[0], line_color=colors_for_curves[0])
        data_for_brush = chart_df[['index', curves[0]]]
    elif len(curves) == 2:
        curve1 = draw_curve(
            chart_df, curves[0], line_color=colors_for_curves[0])
        curve2 = draw_curve(
            chart_df, curves[1], line_color=colors_for_curves[1])
        curve_plots = curve1 + curve2
        data_for_brush = chart_df[['index', curves[0], curves[1]]]
    elif len(curves) == 3:
        curve1 = draw_curve(
            chart_df, curves[0], line_color=colors_for_curves[0])
        curve2 = draw_curve(
            chart_df, curves[1], line_color=colors_for_curves[1])
        curve3 = draw_curve(
            chart_df, curves[2], line_color=colors_for_curves[2])
        curve_plots = curve1 + curve2 + curve3
        data_for_brush = chart_df[['index', curves[0], curves[1], curves[2]]]
    elif len(curves) == 4:
        curve1 = draw_curve(
            chart_df, curves[0], line_color=colors_for_curves[0])
        curve2 = draw_curve(
            chart_df, curves[1], line_color=colors_for_curves[1])
        curve3 = draw_curve(
            chart_df, curves[2], line_color=colors_for_curves[2])
        curve4 = draw_curve(
            chart_df, curves[3], line_color=colors_for_curves[3])
        curve_plots = curve1 + curve2 + curve3 + curve4
        data_for_brush = chart_df[[
            'index', curves[0], curves[1], curves[2], curves[3]
        ]]

    melted_for_brush = pd.melt(data_for_brush, id_vars='index')

    brush_chart = alt.Chart(
        melted_for_brush, height=25, width=chart_width).mark_line(
            color='#d8d8d8', size=1, clip=True).encode(
                x=alt.X('index:T', axis=alt.Axis(title="brush")),
                y=alt.Y('value:Q', axis=alt.Axis(title=None, labelOpacity=0)),
                row=alt.Row(
                    'variable:N',
                    header=alt.Header(labelAngle=0,
                                      labelOrient='top'))).add_selection(brush)

    con_date = conditioning_date

    chart_df['ruler'] = con_date

    vertical_line = alt.Chart(chart_df).mark_rule(
        color='white', size=2).encode(
            x=alt.X('ruler:T', axis=alt.Axis(title=None, labels=False)))

    # dataframe only for purposes of annotations
    anno_df = chart_df.loc[chart_df['index'] == con_date]

    # conditioning date annotation at top of chart
    anno = alt.Chart(anno_df).mark_text(
        align='left',
        baseline='middle',
        size=12,
        color='black',
        fontWeight=300,
        clip=False).encode(
            x=alt.X('ruler:T', scale=alt.Scale(domain=brush)),
            y=alt.value(-5),
            text=alt.value("conditioning date"))

    plot = alt.vconcat(chart + vertical_line + curve_plots + anno, brush_chart)

    if save_as_html:
        plot.save('chart_status.html')

    return plot
