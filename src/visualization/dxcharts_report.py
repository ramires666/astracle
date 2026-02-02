"""
Generate an HTML report with TradingView Lightweight Charts.

NOTE: Verify CDN/version against the latest docs when updating.
Project requirement: use Context7 for latest docs if available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


DEFAULT_TV_VERSION = "5.1.0"


def write_dxcharts_report(
    df: pd.DataFrame,
    output_html: Path,
    title: str = "Labels Report",
    tv_version: Optional[str] = None,
    extra_images: Optional[list[Path]] = None,
    sample_every_days: Optional[int] = None,
    max_points: Optional[int] = None,
) -> Path:
    """
    Generate an HTML page with interactive charts:
    - price + Gaussian (if present)
    - label bars
    - class distribution
    """
    tv_version = tv_version or DEFAULT_TV_VERSION

    # Optional sampling to keep HTML lightweight
    data = df.copy()
    if sample_every_days and int(sample_every_days) > 1 and "date" in data.columns:
        dt = pd.to_datetime(data["date"])
        mask = ((dt - dt.min()).dt.days % int(sample_every_days)) == 0
        data = data.loc[mask].copy()
    if max_points and int(max_points) > 0 and len(data) > int(max_points):
        idx = np.linspace(0, len(data) - 1, int(max_points), dtype=int)
        data = data.iloc[idx].copy()

    # Convert dates to strings so JS reads them correctly
    if "date" in data.columns:
        data["date"] = data["date"].astype(str)
    if "time" in data.columns:
        data["time"] = data["time"].astype(str)

    json_data = data.to_dict(orient="records")
    has_smoothed = "smoothed_close" in data.columns and data["smoothed_close"].notna().any()

    extra_html = ""
    if extra_images:
        img_blocks: list[str] = []
        for img in extra_images:
            try:
                img_path = Path(img)
                try:
                    rel = img_path.relative_to(Path(output_html).parent)
                    src = rel.as_posix()
                except ValueError:
                    src = img_path.as_posix()
                img_blocks.append(
                    f"<div class=\"img-block\"><img src=\"{src}\" alt=\"{img_path.name}\"></div>"
                )
            except Exception:
                continue
        if img_blocks:
            extra_html = "<div class=\"img-wrap\">" + "".join(img_blocks) + "</div>"

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@{tv_version}/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .hint {{ color: #666; font-size: 12px; margin: 6px 0 12px; }}
    #chart_price_wrap {{ position: relative; height: 440px; }}
    #chart_price {{ height: 100%; }}
    #chart_price_zones {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }}
    #chart_labels {{ height: 220px; margin-top: 14px; }}
    #dist {{ margin-top: 14px; }}
    .dist-row {{ display: flex; align-items: center; margin: 6px 0; }}
    .dist-label {{ width: 90px; font-weight: 600; font-size: 12px; }}
    .dist-bar-wrap {{ flex: 1; height: 10px; background: #f0f0f0; margin: 0 10px; position: relative; }}
    .dist-bar {{ height: 10px; }}
    .dist-pct {{ width: 70px; font-size: 12px; color: #444; text-align: right; }}
    .img-wrap {{ margin-top: 18px; }}
    .img-block {{ margin-bottom: 12px; }}
    .img-block img {{ width: 100%; height: auto; border: 1px solid #e0e0e0; }}
    .tv-attrib {{ margin-top: 12px; font-size: 11px; color: #666; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div class="hint">Drag to pan, mouse wheel to zoom.</div>
  <div id="chart_price_wrap">
    <div id="chart_price"></div>
    <canvas id="chart_price_zones"></canvas>
  </div>
  <div id="chart_labels"></div>
  <div id="dist"></div>
  {extra_html}
  <div class="tv-attrib">Charts by TradingView Lightweight Charts</div>

  <script>
    const raw = {json.dumps(json_data)};
    const data = raw.map(d => ({{ ...d, time: d.date }}));
    const targetVals = data.map(d => d.target).filter(v => v !== null && v !== undefined);
    const uniqueLabels = Array.from(new Set(targetVals)).sort((a, b) => a - b);
    const isBinary = uniqueLabels.length === 2 && uniqueLabels.includes(0) && uniqueLabels.includes(1) && !uniqueLabels.includes(2);
    const labelNames = isBinary ? {{ 0: "DOWN", 1: "UP" }} : {{ 0: "DOWN", 1: "SIDEWAYS", 2: "UP" }};
    const labelColors = isBinary
      ? {{ 0: "#d62728", 1: "#2ca02c" }}
      : {{ 0: "#d62728", 1: "#7f7f7f", 2: "#2ca02c" }};
    const labelFills = isBinary
      ? {{ 0: "rgba(214,39,40,0.12)", 1: "rgba(44,160,44,0.12)" }}
      : {{ 0: "rgba(214,39,40,0.12)", 1: "rgba(127,127,127,0.12)", 2: "rgba(44,160,44,0.12)" }};
    const zones = [];
    if (data.length > 0) {{
      let current = data[0].target;
      let start = data[0].time;
      for (let i = 1; i < data.length; i++) {{
        const row = data[i];
        const nextTarget = row.target;
        if (nextTarget !== current) {{
          zones.push({{ start, end: data[i - 1].time, label: current }});
          start = row.time;
          current = nextTarget;
        }}
      }}
      zones.push({{ start, end: data[data.length - 1].time, label: current }});
    }}

    const priceData = data
      .filter(d => d.close !== null && d.close !== undefined)
      .map(d => ({{ time: d.time, value: d.close }}));

    const smoothData = data
      .filter(d => d.smoothed_close !== null && d.smoothed_close !== undefined)
      .map(d => ({{ time: d.time, value: d.smoothed_close }}));

    const labelData = data
      .filter(d => d.target !== null && d.target !== undefined)
      .map(d => ({{ time: d.time, value: 1, color: labelColors[d.target] || "#7f7f7f" }}));

    let priceChart = null;
    if (window.LightweightCharts && typeof window.LightweightCharts.createChart === "function") {{
      priceChart = LightweightCharts.createChart(
        document.getElementById("chart_price"),
        {{
          autoSize: true,
          layout: {{ background: {{ type: "solid", color: "#ffffff" }}, textColor: "#333" }},
          grid: {{ vertLines: {{ color: "#efefef" }}, horzLines: {{ color: "#efefef" }} }},
          rightPriceScale: {{ scaleMargins: {{ top: 0.1, bottom: 0.15 }} }},
          timeScale: {{ timeVisible: true, secondsVisible: false }},
          crosshair: {{ mode: 0 }},
          handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true }},
          handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }}
        }}
      );
    }} else {{
      document.getElementById("chart_price").innerHTML =
        "<div style='color:#b00; padding:12px; font-size:14px;'>TradingView Lightweight Charts failed to load (network). Please refresh.</div>";
    }}

    if (priceChart && typeof priceChart.addLineSeries === "function") {{
      const priceSeries = priceChart.addLineSeries({{
        color: "#111",
        lineWidth: 1,
        pointMarkersVisible: false,
        lastValueVisible: true
      }});
      priceSeries.setData(priceData);
    }}

    if (priceChart && smoothData.length > 0) {{
      const smoothSeries = priceChart.addLineSeries({{
        color: "#1f77b4",
        lineWidth: 1,
        pointMarkersVisible: false,
        lastValueVisible: false
      }});
      smoothSeries.setData(smoothData);
    }}

    let labelChart = null;
    if (priceChart && typeof LightweightCharts !== "undefined") {{
      labelChart = LightweightCharts.createChart(
        document.getElementById("chart_labels"),
        {{
          autoSize: true,
          layout: {{ background: {{ type: "solid", color: "#ffffff" }}, textColor: "#333" }},
          grid: {{ vertLines: {{ color: "#efefef" }}, horzLines: {{ color: "#efefef" }} }},
          rightPriceScale: {{ scaleMargins: {{ top: 0.25, bottom: 0.25 }} }},
          timeScale: {{ timeVisible: true, secondsVisible: false }},
          crosshair: {{ mode: 0 }},
          handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true }},
          handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }}
        }}
      );
    }} else {{
      document.getElementById("chart_labels").innerHTML = "";
    }}

    if (labelChart && typeof labelChart.addHistogramSeries === "function") {{
      const labelSeries = labelChart.addHistogramSeries({{
        priceLineVisible: false,
        lastValueVisible: false
      }});
      labelSeries.setData(labelData);
    }}

    if (priceChart && labelChart) {{
      let syncing = false;
      function syncRanges(source, target) {{
        source.timeScale().subscribeVisibleTimeRangeChange(range => {{
          if (syncing || !range) return;
          syncing = true;
          target.timeScale().setVisibleRange(range);
          syncing = false;
        }});
      }}
      syncRanges(priceChart, labelChart);
      syncRanges(labelChart, priceChart);
    }}

    let drawZones = () => {{}};
    let resizeOverlay = () => {{}};
    if (priceChart) {{
      const chartWrap = document.getElementById("chart_price_wrap");
      const zoneCanvas = document.getElementById("chart_price_zones");
      const zoneCtx = zoneCanvas.getContext("2d");

      resizeOverlay = () => {{
        const rect = chartWrap.getBoundingClientRect();
        zoneCanvas.width = rect.width;
        zoneCanvas.height = rect.height;
        zoneCanvas.style.width = `${{rect.width}}px`;
        zoneCanvas.style.height = `${{rect.height}}px`;
      }}

      drawZones = () => {{
        zoneCtx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);
        if (!zones.length) {{
          return;
        }}
        const timeScale = priceChart.timeScale();
        zones.forEach(zone => {{
          const startCoord = timeScale.timeToCoordinate(zone.start);
          const endCoord = timeScale.timeToCoordinate(zone.end);
          if (
            startCoord === undefined ||
            endCoord === undefined ||
            Number.isNaN(startCoord) ||
            Number.isNaN(endCoord)
          ) {{
            return;
          }}
          const x1 = Math.max(0, Math.min(zoneCanvas.width, startCoord));
          const x2 = Math.max(0, Math.min(zoneCanvas.width, endCoord));
          const x = Math.min(x1, x2);
          const width = Math.max(0, Math.abs(x2 - x1));
          if (width <= 0) {{
            return;
          }}
          zoneCtx.fillStyle = labelFills[zone.label] || "rgba(200,200,200,0.12)";
          zoneCtx.fillRect(x, 0, width, zoneCanvas.height);
        }});
      }}

      if (typeof ResizeObserver !== "undefined") {{
        new ResizeObserver(() => {{
          resizeOverlay();
          drawZones();
        }}).observe(chartWrap);
      }} else {{
        window.addEventListener("resize", () => {{
          resizeOverlay();
          drawZones();
        }});
      }}

      priceChart.timeScale().fitContent();
      resizeOverlay();
      drawZones();
      priceChart.timeScale().subscribeVisibleTimeRangeChange(() => drawZones());
    }}

    if (labelChart) {{
      labelChart.timeScale().fitContent();
      labelChart.timeScale().subscribeVisibleTimeRangeChange(() => {{
        if (priceChart && typeof priceChart.timeScale === "function") {{
          priceChart.timeScale().subscribeVisibleTimeRangeChange(() => drawZones());
        }}
      }});
    }}

    const distEl = document.getElementById("dist");
    if (distEl && targetVals.length > 0) {{
      const total = targetVals.length;
      const counts = {{}};
      targetVals.forEach(v => {{ counts[v] = (counts[v] || 0) + 1; }});
      uniqueLabels.forEach(v => {{
        const pct = (counts[v] || 0) / total * 100;
        const row = document.createElement("div");
        row.className = "dist-row";
        row.innerHTML =
          `<div class="dist-label">${{labelNames[v] || ("Class " + v)}}</div>` +
          `<div class="dist-bar-wrap"><div class="dist-bar" style="width:${{pct.toFixed(2)}}%; background:${{labelColors[v] || "#7f7f7f"}};"></div></div>` +
          `<div class="dist-pct">${{pct.toFixed(2)}}%</div>`;
        distEl.appendChild(row);
      }});
    }}
  </script>
</body>
</html>
"""

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html
