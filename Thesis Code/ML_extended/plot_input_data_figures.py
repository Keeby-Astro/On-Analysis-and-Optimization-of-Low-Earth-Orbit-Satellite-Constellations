from pathlib import Path
import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# UPDATE FIGURE SETTINGS & DEFINE CUSTOM PALETTE
plt.rcParams.update({'figure.figsize': (10.0, 3.75),
                        'xtick.direction': 'in', 'xtick.labelsize': 14, 'xtick.major.size': 3,
                        'xtick.major.width': 0.5, 'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5,
                        'xtick.minor.visible': True, 'xtick.top': True,
                        'ytick.direction': 'in', 'ytick.labelsize': 14, 'ytick.major.size': 3,
                        'ytick.major.width': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
                        'ytick.minor.visible': True, 'ytick.right': True,
                        'axes.linewidth': 0.5, 'grid.linewidth': 0.5, 'lines.linewidth': 1.0,
                        'legend.fontsize': 14, 'legend.frameon': False,
                        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'dejavuserif',
                        'font.size': 12, 'axes.labelsize': 16, 'axes.titlesize': 18,
                        'axes.grid': True, 'grid.linestyle': '--', 'grid.color': '0.5',
                        'lines.markersize': 8, 'axes.spines.top': True, 'axes.spines.right': True})

# Define the custom 20-color palette (darkened colors)
colors = ['#15528e', '#b25800', '#1e701e', '#951c1c', '#673284',
          '#623c34', '#9e5387', '#585858', '#848417', '#108590',
          '#798ba2', '#b28254', '#6a9c60', '#b26a68', '#8a7b94',
          '#896d67', '#ac7f93', '#8b8b8b', '#999962', '#6f989f']


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "input_data_figures"
DEFAULT_SUNSPOT_FILE = "SN_d_tot_V2.0.csv"
DEFAULT_SUNSPOT_START_DATE = "1957-01-01"


def _clean_columns(dataframe):
    dataframe.columns = [str(column).strip().replace('"', '') for column in dataframe.columns]
    return dataframe


def _to_numeric(dataframe, columns):
    for column in columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def load_sw_all(path):
    dataframe = pd.read_csv(path, skipinitialspace=True)
    dataframe = _clean_columns(dataframe)
    if "DATE" not in dataframe.columns:
        raise ValueError(f"{path} must contain a DATE column")

    numeric_columns = [
        "KP_SUM", "AP_AVG", "ISN", "F10.7_OBS", "F10.7_ADJ",
    ]
    numeric_columns.extend([f"KP{index}" for index in range(1, 9)])
    numeric_columns.extend([f"AP{index}" for index in range(1, 9)])

    dataframe["date"] = pd.to_datetime(dataframe["DATE"], errors="coerce")
    dataframe = _to_numeric(dataframe, numeric_columns)
    dataframe = dataframe.dropna(subset=["date"]).sort_values("date")
    return dataframe


def load_sunspot_number(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file_handle:
        first_line = file_handle.readline()

    comma_header_names = ["Year", "Month", "Day", "Fraction", "Total", "Observations", "Indicator"]

    if ";" in first_line and first_line.count(";") >= 6:
        dataframe = pd.read_csv(
            path,
            sep=";",
            engine="python",
            header=None,
            names=["Year", "Month", "Day", "Fraction", "Total", "Observations", "Std", "Indicator"],
        )
    elif first_line.strip().startswith('"Year,'):
        dataframe = pd.read_csv(
            path,
            sep=",",
            engine="python",
            header=None,
            skiprows=1,
            names=comma_header_names,
            skipinitialspace=True,
        )
    else:
        dataframe = pd.read_csv(path, sep=",", engine="python", skipinitialspace=True)
        dataframe = _clean_columns(dataframe)

    required_columns = ["Year", "Month", "Day", "Total"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    dataframe = _to_numeric(dataframe, required_columns)
    dataframe = dataframe.dropna(subset=["Year", "Month", "Day"])
    dataframe["date"] = pd.to_datetime(
        dict(
            year=dataframe["Year"].astype(int),
            month=dataframe["Month"].astype(int),
            day=dataframe["Day"].astype(int),
        ),
        errors="coerce",
    )
    dataframe["Total"] = dataframe["Total"].where(dataframe["Total"] >= 0, np.nan)
    dataframe = dataframe.dropna(subset=["date"]).sort_values("date")
    return dataframe[dataframe["date"] >= pd.Timestamp(DEFAULT_SUNSPOT_START_DATE)]


def load_dst_index(path):
    dataframe = pd.read_csv(path, skipinitialspace=True)
    dataframe = _clean_columns(dataframe)

    date_column = "Date" if "Date" in dataframe.columns else "DATE"
    if date_column not in dataframe.columns:
        raise ValueError(f"{path} must contain a Date column")

    dst_column = next((column for column in dataframe.columns if column.upper() == "DST"), None)
    if dst_column is None:
        raise ValueError(f"{path} must contain a DST column")

    dataframe["date"] = pd.to_datetime(dataframe[date_column], errors="coerce")
    dataframe[dst_column] = pd.to_numeric(dataframe[dst_column], errors="coerce")
    dataframe = dataframe.dropna(subset=["date"]).sort_values("date")
    return dataframe.rename(columns={dst_column: "DST"})


def daily_kp(sw_all):
    if "KP_SUM" in sw_all.columns and sw_all["KP_SUM"].notna().any():
        kp = sw_all["KP_SUM"] / 80.0
    else:
        kp_columns = [f"KP{index}" for index in range(1, 9) if f"KP{index}" in sw_all.columns]
        if not kp_columns:
            raise ValueError("SW-All data must contain KP_SUM or KP1-KP8 columns")
        kp = sw_all[kp_columns].mean(axis=1) / 10.0
    return kp.clip(lower=0, upper=9)


def daily_ap(sw_all):
    if "AP_AVG" in sw_all.columns and sw_all["AP_AVG"].notna().any():
        return sw_all["AP_AVG"].clip(lower=0, upper=400)

    ap_columns = [f"AP{index}" for index in range(1, 9) if f"AP{index}" in sw_all.columns]
    if not ap_columns:
        raise ValueError("SW-All data must contain AP_AVG or AP1-AP8 columns")
    return sw_all[ap_columns].mean(axis=1).clip(lower=0, upper=400)


def format_time_axis(axis):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.tick_params(which="both", top=True, right=True)
    axis.minorticks_on()


def save_plot(figure, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=600, bbox_inches="tight")
    return path


def plot_solar_radio_flux(sw_all, output_dir):
    figure, axis = plt.subplots()
    ursi_d = sw_all["F10.7_ADJ"] * 0.9

    axis.plot(sw_all["date"], sw_all["F10.7_OBS"], color=colors[0], label="Observed Solar Radio Flux")
    axis.plot(sw_all["date"], sw_all["F10.7_ADJ"], color=colors[1], label="Adjusted Solar Radio Flux")
    axis.plot(sw_all["date"], ursi_d, color=colors[2], label="URSI-D Solar Radio Flux")

    axis.set_xlabel("Date")
    axis.set_ylabel("10.7 cm Solar Radio Flux (sfu)")
    axis.legend(loc="best")
    format_time_axis(axis)
    figure.tight_layout()
    return save_plot(figure, output_dir, "solar_radio_flux_f107.png")


def plot_kp(sw_all, output_dir):
    figure, axis = plt.subplots()
    axis.plot(sw_all["date"], daily_kp(sw_all), color=colors[3], label="Daily mean Kp")
    axis.set_xlabel("Date")
    axis.set_ylabel("Kp Index")
    axis.set_ylim(0, 9)
    format_time_axis(axis)
    figure.tight_layout()
    return save_plot(figure, output_dir, "kp_index.png")


def plot_ap(sw_all, output_dir):
    figure, axis = plt.subplots()
    axis.plot(sw_all["date"], daily_ap(sw_all), color=colors[4], label="Daily mean ap")
    axis.set_xlabel("Date")
    axis.set_ylabel("ap Index")
    format_time_axis(axis)
    figure.tight_layout()
    return save_plot(figure, output_dir, "ap_index.png")


def plot_sunspot(sunspot, output_dir):
    figure, axis = plt.subplots()
    axis.plot(sunspot["date"], sunspot["Total"], color=colors[5], label="Daily sunspot number")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sunspot Number")
    format_time_axis(axis)
    figure.tight_layout()
    return save_plot(figure, output_dir, "sunspot_number.png")


def plot_dst(dst, output_dir):
    figure, axis = plt.subplots()
    axis.plot(dst["date"], dst["DST"], color=colors[6], label="Daily Dst")
    axis.axhline(0, color=colors[7], linewidth=0.8)
    axis.set_xlabel("Date")
    axis.set_ylabel("Dst Index (nT)")
    format_time_axis(axis)
    figure.tight_layout()
    return save_plot(figure, output_dir, "dst_index.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ML_extended input data figures.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing SW-All.csv and related input data files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder where plot images will be saved.")
    parser.add_argument("--show", action="store_true", help="Display the figures after saving them.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    sw_all = load_sw_all(data_dir / "SW-All.csv")
    sunspot = load_sunspot_number(data_dir / DEFAULT_SUNSPOT_FILE)
    dst = load_dst_index(data_dir / "dst_index_daily.csv")

    saved_paths = [
        plot_solar_radio_flux(sw_all, output_dir),
        plot_kp(sw_all, output_dir),
        plot_ap(sw_all, output_dir),
        plot_sunspot(sunspot, output_dir),
        plot_dst(dst, output_dir),
    ]

    print("Saved input data figures:")
    for path in saved_paths:
        print(f"  {path}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()