import os
from datetime import datetime, timedelta
from pathlib import Path
import html as html_lib
import webbrowser

import pandas as pd
import plotly.express as px
import plotly.io as pio
from load_all_tle_data import load_all_tle_data

def _as_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')

def _tle_epoch_str_from_timestamp(ts: pd.Timestamp) -> str:
    tsu = _as_utc(ts)
    yy = tsu.year % 100
    doy = int(tsu.strftime('%j'))
    seconds = tsu.hour * 3600 + tsu.minute * 60 + tsu.second + (tsu.microsecond / 1e6)
    frac = seconds / 86400.0

    frac_digits = int(round(frac * 1e8))
    if frac_digits >= 100000000:
        # Handle rare rounding carry at end-of-day
        frac_digits = 0
        doy += 1
        # If we carry beyond the year, roll day-of-year and year
        year_days = 366 if tsu.is_leap_year else 365
        if doy > year_days:
            doy = 1
            yy = (yy + 1) % 100

    return f"{yy:02d}{doy:03d}.{frac_digits:08d}"

def _utc_ymdhms_str(ts: pd.Timestamp) -> str:
    tsu = _as_utc(ts)
    return (f"{tsu.year:04d}/{tsu.month:02d}/{tsu.day:02d}/"
            f"{tsu.hour:02d}/{tsu.minute:02d}/{tsu.second:02d}."
            f"{tsu.microsecond // 100:04d}")


def _timestamp_from_tle_epoch_str(tle_epoch: str) -> pd.Timestamp:
    """Convert TLE epoch string 'YYDDD.DDDDDDDD' to a UTC pandas Timestamp."""
    s = str(tle_epoch).strip()
    yyddd = float(s)
    yy = int(yyddd // 1000)
    ddd = yyddd - yy * 1000
    year = 2000 + yy if yy < 57 else 1900 + yy
    epoch_dt = datetime(year, 1, 1) + timedelta(days=ddd - 1)
    return pd.Timestamp(epoch_dt, tz='UTC')


def _write_decay_copy(*, src_path: str, dst_path: str, start_epoch: str) -> int:
    """Write a filtered TLE copy containing blocks with epoch >= start_epoch.

    Returns number of TLE blocks written.
    """
    start_ts = _timestamp_from_tle_epoch_str(start_epoch)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    blocks_written = 0

    def _iter_nonblank_lines(fh):
        for raw in fh:
            ln = raw.rstrip('\n')
            if ln.strip():
                yield ln.rstrip('\r')

    with open(src_path, 'r') as fin, open(dst_path, 'w', newline='\n') as fout:
        it = _iter_nonblank_lines(fin)
        for name, line1, line2 in zip(it, it, it):
            if len(line1) < 32:
                continue
            epoch_str = line1[18:32].strip()
            if not epoch_str:
                continue
            try:
                ts = _timestamp_from_tle_epoch_str(epoch_str)
            except Exception:
                continue

            if ts >= start_ts:
                fout.write(f"{name}\n{line1}\n{line2}\n")
                blocks_written += 1

    return blocks_written


def _write_plotly_click_copy_html(*, fig, out_html_path: str, div_id: str,
                                  default_box_text: str) -> str:
        """Write a Plotly HTML with a persistent selection box.

        Clicking a point copies the TLE epoch (customdata[0]) to clipboard and pins
        the selected point details in a box above the plot.

        Returns the absolute path written.
        """
        out_path = Path(out_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig.update_layout(hovermode='closest', clickmode='event+select')

        # NOTE: Plotly inserts this script into the generated HTML (after graph creation).
        post_script = f"""
        (function() {{
            const gd = document.getElementById('{div_id}');
            const selectedBox = document.getElementById('selected-box');
            const jsWarning = document.getElementById('js-warning');
            if (!gd || !selectedBox) return;

            if (jsWarning) {{
                jsWarning.style.display = 'none';
            }}

            function setText(t) {{
                selectedBox.textContent = t;
            }}

            function copyText(t) {{
                if (!t) return;
                try {{
                    if (navigator.clipboard && window.isSecureContext) {{
                        navigator.clipboard.writeText(t);
                        return;
                    }}
                }} catch (e) {{ /* fall back */ }}

                const ta = document.createElement('textarea');
                ta.value = t;
                ta.setAttribute('readonly', '');
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                ta.style.top = '0';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                try {{ document.execCommand('copy'); }} catch (e) {{ /* ignore */ }}
                document.body.removeChild(ta);
            }}

            gd.on('plotly_click', function(evt) {{
                if (!evt || !evt.points || !evt.points.length) return;
                const p = evt.points[0];
                const cd = p.customdata || [];
                const tleEpoch = (cd.length > 0) ? cd[0] : '';
                const utc = (cd.length > 1) ? cd[1] : '';
                const y = (typeof p.y !== 'undefined') ? p.y : '';
                const x = (typeof p.x !== 'undefined') ? p.x : '';

                const text = `TLE Epoch: ${{tleEpoch}}
UTC: ${{utc}}
X: ${{x}}
SMA: ${{y}}`;
                setText(text);
                copyText(tleEpoch);
            }});
        }})();
        """

        plot_fragment = pio.to_html(fig, full_html=False, include_plotlyjs=True,
                                    div_id=div_id, post_script=post_script,
                                    default_width='100%', default_height='700px')
        header_text = html_lib.escape(default_box_text)
        full_html = (
                "<!doctype html>\n"
                "<html lang='en'>\n"
                "  <head>\n"
                "    <meta charset='utf-8'/>\n"
                "    <meta name='viewport' content='width=device-width, initial-scale=1'/>\n"
                "    <title>Starlink plot</title>\n"
                "  </head>\n"
                "  <body>\n"
            "    <div id='js-warning' style='font-family: sans-serif; margin: 12px 8px; border: 1px solid #bbb; padding: 8px;'>\n"
            "      This interactive plot requires JavaScript. If you see a blank plot area, you're viewing it in a restricted preview (often VS Code) that blocks scripts.\n"
            "      Open this HTML in a regular browser (Edge/Chrome/Firefox).\n"
            "    </div>\n"
                "    <div style='font-family: sans-serif; margin: 12px 8px;'>\n"
                "      <div style='margin-bottom: 6px;'><b>Click a point</b> to copy the TLE epoch to clipboard. The selected point stays pinned below.</div>\n"
                "      <pre id='selected-box' style='white-space: pre-wrap; border: 1px solid #bbb; padding: 8px; margin: 0;'>"
                + header_text +
                "</pre>\n"
                "    </div>\n"
                + plot_fragment +
                "  </body>\n"
                "</html>\n"
        )

        out_path.write_text(full_html, encoding='utf-8')
        return str(out_path.resolve())

def _iter_satellite_numbers_from_file(path: str):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            yield s


# Load and preprocess TLE data for STARLINK
folder_path = r'C:\Users\PC\Code\starlink_tles'
plot_out_dir = r'C:\Users\PC\Code\starlink_decay_plots'
sat_list_path = r'C:\Users\PC\Code\sat_list.txt'

satellite_numbers = list(_iter_satellite_numbers_from_file(sat_list_path))

for i, satellite_number in enumerate(satellite_numbers, start=1):
    print(f"\n[{i}/{len(satellite_numbers)}] sat{satellite_number}")

    src_filename = f"sat{satellite_number}.txt"
    src_path = os.path.join(folder_path, src_filename)
    if not os.path.exists(src_path):
        print(f"  Skipping: not found: {src_path}")
        continue

    df, _ = load_all_tle_data([folder_path], only_files=[src_filename], derived={'sma'})
    if df is None or len(df) == 0:
        print("  Skipping: no data loaded")
        continue

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['tle_epoch'] = df['timestamp'].apply(_tle_epoch_str_from_timestamp)
    df['utc_ymdhms'] = df['timestamp'].apply(_utc_ymdhms_str)

    # Semi-major axis plot
    fig = px.scatter(
        df,
        x='timestamp',
        y='sma',
        custom_data=['tle_epoch', 'utc_ymdhms'],
        title=f'Starlink {satellite_number}: Semi-major Axis vs. Epoch (UTC)',
        labels={'timestamp': 'Epoch (UTC)', 'sma': 'Semi-major Axis (km)'},
    )

    fig.update_traces(
        marker={'size': 5},
        mode='lines+markers',
        hovertemplate=(
            'TLE Epoch: %{customdata[0]}<br>'
            'UTC: %{customdata[1]}<br>'
            'SMA: %{y:.3f} km'
            '<extra></extra>'
        ),
    )

    fig.update_layout(xaxis={'tickangle': 45})

    main_html = os.path.join(plot_out_dir, f"sat{satellite_number}_sma.html")
    main_abs = _write_plotly_click_copy_html(
        fig=fig,
        out_html_path=main_html,
        div_id=f"sat{satellite_number}_main_plot",
        default_box_text="Click a point to copy its TLE Epoch.",
    )
    webbrowser.open(Path(main_abs).as_uri())

    # -------------------------------------------------------------------------
    # Create a filtered "decay" copy starting from a chosen TLE epoch
    # -------------------------------------------------------------------------
    start_epoch = input(
        f"Enter starting TLE epoch (YYDDD.DDDDDDDD) to create decay copy for sat{satellite_number} (or blank to skip): "
    ).strip()

    if not start_epoch:
        continue

    decay_folder = r'C:\Users\PC\Code\starlink_decay'
    dst_file = os.path.join(decay_folder, f"sat{satellite_number}_decay.txt")

    written = _write_decay_copy(src_path=src_path, dst_path=dst_file, start_epoch=start_epoch)
    if written <= 0:
        raise RuntimeError(
            "No TLE blocks were written. Check that the epoch exists in the file, "
            "and that the starting epoch is not after the last entry."
        )

    df_decay, _ = load_all_tle_data(
        [decay_folder],
        only_files=[f"sat{satellite_number}_decay.txt"],
        derived={'sma'},
    )
    if df_decay is None or len(df_decay) == 0:
        print("  Skipping decay plot: no data loaded")
        continue

    df_decay['timestamp'] = pd.to_datetime(df_decay['timestamp'])
    df_decay.sort_values('timestamp', inplace=True)
    df_decay.reset_index(drop=True, inplace=True)

    df_decay['tle_epoch'] = df_decay['timestamp'].apply(_tle_epoch_str_from_timestamp)
    df_decay['utc_ymdhms'] = df_decay['timestamp'].apply(_utc_ymdhms_str)

    fig2 = px.scatter(
        df_decay,
        x='timestamp',
        y='sma',
        custom_data=['tle_epoch', 'utc_ymdhms'],
        title=f"Starlink {satellite_number} (Decay Copy): Semi-major Axis vs. Epoch (UTC)",
        labels={'timestamp': 'Epoch (UTC)', 'sma': 'Semi-major Axis (km)'},
    )

    fig2.update_traces(
        marker={'size': 5},
        mode='lines+markers',
        hovertemplate=(
            'TLE Epoch: %{customdata[0]}<br>'
            'UTC: %{customdata[1]}<br>'
            'SMA: %{y:.3f} km'
            '<extra></extra>'
        ),
    )
    fig2.update_layout(xaxis={'tickangle': 45})

    decay_html = os.path.join(plot_out_dir, f"sat{satellite_number}_decay_sma.html")
    decay_abs = _write_plotly_click_copy_html(
        fig=fig2,
        out_html_path=decay_html,
        div_id=f"sat{satellite_number}_decay_plot",
        default_box_text=f"Decay copy start epoch: {start_epoch}",
    )
    webbrowser.open(Path(decay_abs).as_uri())