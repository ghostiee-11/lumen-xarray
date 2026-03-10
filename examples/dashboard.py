"""
lumen-xarray Dashboard — Interactive explorer for any xarray dataset.

Run:
    PYTHONPATH=. panel serve examples/dashboard.py --show
"""

import io
import sys
import tempfile
import argparse

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
import xarray as xr
import hvplot.pandas  # noqa: F401 — registers hvplot accessor

pn.extension("tabulator", notifications=True)
hv.extension("bokeh")

from lumen_xarray import XArraySQLSource
from lumen_xarray.source import XARRAY_EXTENSIONS
from lumen_xarray.transforms import (
    DimensionSlice, SpatialBBox, DimensionAggregate,
    TimeResample, Anomaly, RollingWindow,
)


# ────────────────────────────────────────────────────────────
# State: holds the current source + derived info
# ────────────────────────────────────────────────────────────
class AppState:
    """Mutable application state — replaced when a new dataset is loaded."""

    def __init__(self, source, ds, name):
        self.source = source
        self.ds = ds
        self.name = name
        self.tables = source.get_tables()
        self.primary_var = self.tables[0]
        self.dim_info = source.get_dimension_info(self.primary_var)
        self.size_info = source.estimate_size(self.primary_var)

        # Detect coordinate keys
        self.time_key = next(
            (k for k, v in self.dim_info.items() if v.get("type") == "datetime"), None
        )
        self.lat_key = next(
            (k for k in ("lat", "latitude") if k in self.dim_info), None
        )
        self.lon_key = next(
            (k for k in ("lon", "longitude") if k in self.dim_info), None
        )
        self.has_time = self.time_key is not None
        self.has_spatial = self.lat_key is not None and self.lon_key is not None


# Start with demo data
def _load_demo():
    ds = xr.tutorial.open_dataset("air_temperature")
    src = XArraySQLSource(_dataset=ds)
    return AppState(src, ds, "NOAA Air Temperature (demo)")


state = _load_demo()


# ────────────────────────────────────────────────────────────
# Sidebar: Data loader (file upload + path input)
# ────────────────────────────────────────────────────────────
file_input = pn.widgets.FileInput(
    name="Upload Dataset",
    accept=",".join(XARRAY_EXTENSIONS),
    multiple=False,
    height=60,
)
path_input = pn.widgets.TextInput(
    name="Or enter file path / URL",
    placeholder="/path/to/data.nc  or  s3://bucket/data.zarr",
    width=250,
)
load_btn = pn.widgets.Button(name="Load", button_type="primary", width=250)
load_status = pn.pane.Markdown("", width=250)

# Dynamic control widgets — rebuilt on dataset change
controls_column = pn.Column(width=270)
analysis_widgets_column = pn.Column(width=270)

# These are always present
resample_freq = pn.widgets.Select(
    name="Resample", options=["D", "W", "MS", "QS", "YS"], value="MS",
)
anomaly_groupby = pn.widgets.Select(
    name="Anomaly Baseline", options=["month", "season", "dayofyear", None], value="month",
)
rolling_window = pn.widgets.IntSlider(
    name="Rolling Window", start=1, end=90, value=30,
)

# Dynamic widgets — rebuilt per dataset
time_range = None
lat_range = None
lon_range = None
var_select = None
extra_dim_widgets = {}


def _build_controls():
    """Rebuild sidebar widgets from current state."""
    global time_range, lat_range, lon_range, var_select, extra_dim_widgets

    s = state
    dim_info = s.dim_info
    widget_list = []
    extra_dim_widgets = {}

    # Time
    if s.time_key and dim_info[s.time_key].get("min"):
        t_min = pd.Timestamp(dim_info[s.time_key]["min"])
        t_max = pd.Timestamp(dim_info[s.time_key]["max"])
        time_range = pn.widgets.DateRangeSlider(
            name="Time Range", start=t_min, end=t_max, value=(t_min, t_max),
        )
        widget_list.append(time_range)
    else:
        time_range = None

    # Lat
    if s.lat_key and dim_info[s.lat_key].get("min") is not None:
        li = dim_info[s.lat_key]
        step = abs(li.get("step") or 1.0)
        lat_range = pn.widgets.RangeSlider(
            name="Latitude",
            start=float(li["min"]), end=float(li["max"]),
            value=(float(li["min"]), float(li["max"])),
            step=step,
        )
        widget_list.append(lat_range)
    else:
        lat_range = None

    # Lon
    if s.lon_key and dim_info[s.lon_key].get("min") is not None:
        li = dim_info[s.lon_key]
        step = abs(li.get("step") or 1.0)
        lon_range = pn.widgets.RangeSlider(
            name="Longitude",
            start=float(li["min"]), end=float(li["max"]),
            value=(float(li["min"]), float(li["max"])),
            step=step,
        )
        widget_list.append(lon_range)
    else:
        lon_range = None

    # Extra numeric dims
    for dim_name, dinfo in dim_info.items():
        if dim_name in (s.time_key, s.lat_key, s.lon_key):
            continue
        if dinfo["type"] == "numeric":
            w = pn.widgets.RangeSlider(
                name=dim_name.capitalize(),
                start=float(dinfo["min"]), end=float(dinfo["max"]),
                value=(float(dinfo["min"]), float(dinfo["max"])),
            )
            extra_dim_widgets[dim_name] = w
            widget_list.append(w)

    # Variable selector
    if len(s.tables) > 1:
        var_select = pn.widgets.Select(
            name="Variable", options=s.tables, value=s.primary_var,
        )
        widget_list.append(var_select)
    else:
        var_select = None

    controls_column.clear()
    controls_column.extend(widget_list)

    analysis_widgets_column.clear()
    analysis_widgets_column.extend([resample_freq, anomaly_groupby, rolling_window])


_build_controls()


# ────────────────────────────────────────────────────────────
# Dataset loading logic
# ────────────────────────────────────────────────────────────
def _load_from_path(path):
    global state
    try:
        source = XArraySQLSource(uri=path)
        state = AppState(source, source.dataset, path.split("/")[-1])
        _build_controls()
        _rebuild_tabs()
        load_status.object = f"**Loaded:** {state.name} ({len(state.tables)} vars)"
        pn.state.notifications.success(f"Loaded {state.name}")
    except Exception as e:
        load_status.object = f"**Error:** {e}"
        pn.state.notifications.error(str(e))


def _load_from_upload(event):
    global state
    if file_input.value is None:
        return
    fname = file_input.filename
    suffix = "." + fname.rsplit(".", 1)[-1] if "." in fname else ".nc"
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(file_input.value)
        tmp.flush()
        tmp.close()
        source = XArraySQLSource(uri=tmp.name)
        state = AppState(source, source.dataset, fname)
        _build_controls()
        _rebuild_tabs()
        load_status.object = f"**Loaded:** {state.name} ({len(state.tables)} vars)"
        pn.state.notifications.success(f"Loaded {state.name}")
    except Exception as e:
        load_status.object = f"**Error:** {e}"
        pn.state.notifications.error(str(e))


def _on_load_click(event):
    if file_input.value is not None:
        _load_from_upload(event)
    elif path_input.value.strip():
        _load_from_path(path_input.value.strip())
    else:
        load_status.object = "Upload a file or enter a path first."


load_btn.on_click(_on_load_click)


# ────────────────────────────────────────────────────────────
# Filtering helpers
# ────────────────────────────────────────────────────────────
def _current_var():
    return var_select.value if var_select else state.primary_var


def _get_filtered():
    s = state
    var = _current_var()
    df = s.source.execute(f"SELECT * FROM {var}")

    if time_range is not None:
        df = DimensionSlice(
            dimension=s.time_key,
            start=str(time_range.value[0]),
            stop=str(time_range.value[1]),
        ).apply(df)

    if lat_range is not None and lon_range is not None:
        df = SpatialBBox(
            lat_col=s.lat_key, lon_col=s.lon_key,
            lat_min=lat_range.value[0], lat_max=lat_range.value[1],
            lon_min=lon_range.value[0], lon_max=lon_range.value[1],
        ).apply(df)
    elif lat_range is not None:
        df = DimensionSlice(
            dimension=s.lat_key,
            start=lat_range.value[0], stop=lat_range.value[1],
        ).apply(df)
    elif lon_range is not None:
        df = DimensionSlice(
            dimension=s.lon_key,
            start=lon_range.value[0], stop=lon_range.value[1],
        ).apply(df)

    for dim_name, w in extra_dim_widgets.items():
        df = DimensionSlice(
            dimension=dim_name, start=w.value[0], stop=w.value[1],
        ).apply(df)

    return df


# ────────────────────────────────────────────────────────────
# Apply button: triggers plot updates
# ────────────────────────────────────────────────────────────
apply_btn = pn.widgets.Button(name="Apply Filters", button_type="success", width=270)
_update_counter = pn.widgets.IntInput(value=0, visible=False)  # reactive trigger

def _on_apply(event):
    _update_counter.value += 1

apply_btn.on_click(_on_apply)


# ────────────────────────────────────────────────────────────
# Visualization panels (each returns an hvplot object)
# ────────────────────────────────────────────────────────────
@pn.depends(_update_counter)
def _spatial_heatmap(trigger=None):
    s = state
    df = _get_filtered()
    var = _current_var()
    if df.empty:
        return pn.pane.Markdown("No data for selected range.")

    if s.has_spatial:
        # Aggregate all non-spatial dims (time, level, etc.)
        non_spatial = [d for d in s.dim_info if d not in (s.lat_key, s.lon_key) and d in df.columns]
        agg = DimensionAggregate(
            dimensions=non_spatial, method="mean", value_columns=[var],
        ).apply(df) if non_spatial else df
        return agg.hvplot.heatmap(
            x=s.lon_key, y=s.lat_key, C=var, cmap="RdYlBu_r",
            title=f"Mean {var}", width=650, height=420, colorbar=True,
        )
    elif len(s.dim_info) >= 2:
        dims = [d for d in s.dim_info if d != s.time_key][:2]
        non_plot = [d for d in s.dim_info if d not in dims and d in df.columns]
        agg = DimensionAggregate(
            dimensions=non_plot, method="mean", value_columns=[var],
        ).apply(df) if non_plot else df
        return agg.hvplot.heatmap(
            x=dims[0], y=dims[1], C=var, cmap="viridis",
            title=f"Mean {var}", width=650, height=420,
        )
    return pn.pane.Markdown("Need 2+ spatial dimensions for heatmap.")


@pn.depends(_update_counter, resample_freq)
def _time_series(trigger=None, freq=None):
    s = state
    df = _get_filtered()
    var = _current_var()
    if df.empty or not s.has_time:
        return pn.pane.Markdown("No time dimension available.")

    non_time = [d for d in s.dim_info if d != s.time_key and d in df.columns]
    agg = DimensionAggregate(
        dimensions=non_time, method="mean", value_columns=[var],
    ).apply(df) if non_time else df

    resampled = TimeResample(time_col=s.time_key, freq=resample_freq.value).apply(agg)
    return resampled.hvplot.line(
        x=s.time_key, y=var, color="#2196F3",
        title=f"Spatially-Averaged {var} ({resample_freq.value})",
        width=720, height=370,
    )


@pn.depends(_update_counter, anomaly_groupby)
def _anomaly(trigger=None, groupby=None):
    s = state
    df = _get_filtered()
    var = _current_var()
    if df.empty or not s.has_time:
        return pn.pane.Markdown("No time dimension available.")

    non_time = [d for d in s.dim_info if d != s.time_key and d in df.columns]
    agg = DimensionAggregate(
        dimensions=non_time, method="mean", value_columns=[var],
    ).apply(df) if non_time else df

    anom = Anomaly(
        time_col=s.time_key, value_col=var, groupby=anomaly_groupby.value,
    ).apply(agg)
    anom_col = f"{var}_anomaly"
    colors = ["#EF5350" if v >= 0 else "#42A5F5" for v in anom[anom_col]]
    return anom.hvplot.bar(
        x=s.time_key, y=anom_col, color=colors,
        title=f"{var} Anomaly (baseline: {anomaly_groupby.value or 'overall'})",
        width=720, height=370, rot=45,
    )


@pn.depends(_update_counter, rolling_window)
def _rolling(trigger=None, window=None):
    s = state
    df = _get_filtered()
    var = _current_var()
    if df.empty or not s.has_time:
        return pn.pane.Markdown("No time dimension available.")

    non_time = [d for d in s.dim_info if d != s.time_key and d in df.columns]
    agg = DimensionAggregate(
        dimensions=non_time, method="mean", value_columns=[var],
    ).apply(df) if non_time else df

    smoothed = RollingWindow(column=var, window=rolling_window.value).apply(agg)
    rolling_col = f"{var}_rolling_mean"
    raw = smoothed.hvplot.line(
        x=s.time_key, y=var, label="Raw", color="#BDBDBD", alpha=0.5,
    )
    smooth = smoothed.hvplot.line(
        x=s.time_key, y=rolling_col, color="#FF5722",
        label=f"{rolling_window.value}-day Mean",
    )
    return (raw * smooth).opts(
        title=f"{var} — {rolling_window.value}-day Rolling Mean",
        width=720, height=370, legend_position="top_left",
    )


# ────────────────────────────────────────────────────────────
# SQL Explorer
# ────────────────────────────────────────────────────────────
sql_input = pn.widgets.TextAreaInput(name="SQL Query", height=100, width=650)
sql_run = pn.widgets.Button(name="Run Query", button_type="primary")
sql_status = pn.pane.Markdown("")
sql_result = pn.pane.DataFrame(pd.DataFrame(), width=720)


def _set_default_sql():
    s = state
    dims = list(s.dim_info.keys())[:2]
    sql_input.value = (
        f"SELECT {', '.join(dims)}, AVG({s.primary_var}) as avg_val\n"
        f"FROM {s.primary_var}\n"
        f"GROUP BY {', '.join(dims)}\n"
        f"ORDER BY {dims[0]}"
    )


def _run_sql(event):
    try:
        sql_status.object = "*Running...*"
        df = state.source.execute(sql_input.value)
        sql_result.object = df.head(1000)
        n = len(df)
        sql_status.object = f"**{n:,} rows**" + (" (showing first 1,000)" if n > 1000 else "")
    except Exception as e:
        sql_result.object = pd.DataFrame({"error": [str(e)]})
        sql_status.object = f"**Error:** {e}"


sql_run.on_click(_run_sql)
_set_default_sql()


# ────────────────────────────────────────────────────────────
# Dataset Info pane
# ────────────────────────────────────────────────────────────
info_pane = pn.pane.Markdown("", width=720)


def _build_info():
    s = state
    meta = s.source.get_metadata(s.primary_var)
    md = f"### Dataset: {s.name}\n{meta.get('description', 'N/A')}\n\n"
    md += f"**Variables:** {', '.join(s.tables)}  \n"
    md += f"**Data points:** {s.size_info['rows']:,} ({s.size_info['estimated_mb']} MB est.)\n\n"
    md += "| Dimension | Type | Range | Size |\n|---|---|---|---|\n"
    for dim, d in s.dim_info.items():
        md += f"| {dim} | {d['type']} | {d.get('min', 'N/A')} — {d.get('max', 'N/A')} | {d['size']} |\n"

    if len(s.tables) > 1:
        md += "\n### All Variables\n"
        for tbl in s.tables:
            tbl_meta = s.source.get_metadata(tbl)
            md += f"- **{tbl}**: {tbl_meta.get('description', 'N/A')}\n"

    ds_attrs = dict(s.ds.attrs)
    if ds_attrs:
        md += "\n### Attributes\n"
        for k, v in ds_attrs.items():
            md += f"- **{k}**: {v}\n"

    info_pane.object = md


_build_info()


# ────────────────────────────────────────────────────────────
# Tabs container — rebuilt when dataset changes
# ────────────────────────────────────────────────────────────
main_tabs = pn.Tabs()


def _rebuild_tabs():
    """Rebuild all tabs and info for the current dataset."""
    s = state

    _set_default_sql()
    _build_info()

    main_tabs.clear()
    if s.has_spatial or len(s.dim_info) >= 2:
        main_tabs.append(("Spatial Heatmap", pn.panel(_spatial_heatmap)))
    if s.has_time:
        main_tabs.append(("Time Series", pn.panel(_time_series)))
        main_tabs.append(("Anomaly", pn.panel(_anomaly)))
        main_tabs.append(("Rolling Mean", pn.panel(_rolling)))
    main_tabs.append(("SQL Explorer", pn.Column(sql_input, sql_run, sql_status, sql_result)))
    main_tabs.append(("Dataset Info", info_pane))


_rebuild_tabs()


# ────────────────────────────────────────────────────────────
# Template
# ────────────────────────────────────────────────────────────
sidebar = pn.Column(
    "## Load Data",
    file_input,
    path_input,
    load_btn,
    load_status,
    pn.layout.Divider(),
    "## Filters",
    controls_column,
    apply_btn,
    _update_counter,
    pn.layout.Divider(),
    "## Analysis",
    analysis_widgets_column,
    width=290,
)

template = pn.template.FastListTemplate(
    title="lumen-xarray",
    sidebar=[sidebar],
    main=[main_tabs],
    accent_base_color="#1976D2",
    header_background="#1976D2",
)

# ── CLI arg support (panel serve ... --args file.nc) ──
argv = getattr(pn.state, "argv", None) or sys.argv[1:]
_parser = argparse.ArgumentParser()
_parser.add_argument("dataset", nargs="?", default=None)
_cli, _ = _parser.parse_known_args(argv)
if _cli.dataset:
    _load_from_path(_cli.dataset)

template.servable()
