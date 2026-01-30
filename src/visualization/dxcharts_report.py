"""
Generate an HTML report with charts via dxChart (DevExtreme).

NOTE: Always verify dxChart URL/version against the latest documentation.
Project requirement: use Context7 for the latest docs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DX_VERSION = "23.2.5"  # TODO: update based on latest docs


def write_dxcharts_report(
    df: pd.DataFrame,
    output_html: Path,
    title: str = "Oracle Labels Report",
    dx_version: Optional[str] = None,
) -> Path:
    """
    Generate an HTML page with charts:
    - close
    - smoothed_close
    - target (colored bars)
    """
    dx_version = dx_version or DEFAULT_DX_VERSION

    # Convert dates to strings so JS reads them correctly
    data = df.copy()
    if "date" in data.columns:
        data["date"] = data["date"].astype(str)
    if "time" in data.columns:
        data["time"] = data["time"].astype(str)

    json_data = data.to_dict(orient="records")

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <link rel="stylesheet" href="https://cdn3.devexpress.com/jslib/{dx_version}/css/dx.light.css">
  <script src="https://cdn3.devexpress.com/jslib/{dx_version}/js/dx.all.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    #chart_price {{ height: 420px; }}
    #chart_labels {{ height: 200px; margin-top: 20px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div id="chart_price"></div>
  <div id="chart_labels"></div>

  <script>
    const data = {json.dumps(json_data)};
    const seriesPrice = [
      {{ valueField: "close", argumentField: "date", name: "Close" }},
      {{ valueField: "smoothed_close", argumentField: "date", name: "Smoothed" }}
    ];

    // Price + smoothing chart
    new DevExpress.viz.dxChart("#chart_price", {{
      dataSource: data,
      commonSeriesSettings: {{
        argumentField: "date",
        type: "line"
      }},
      series: seriesPrice,
      legend: {{ visible: true }},
      tooltip: {{ enabled: true }},
      argumentAxis: {{
        valueMarginsEnabled: false
      }},
      // Zoom and pan (mouse/touch)
      zoomAndPan: {{
        argumentAxis: "both",
        valueAxis: "both",
        dragToZoom: true,
        allowMouseWheel: true,
        allowTouchGestures: true,
        panKey: "shift"
      }}
    }});

    // Target labels (0/1/2) as bars
    new DevExpress.viz.dxChart("#chart_labels", {{
      dataSource: data,
      commonSeriesSettings: {{
        argumentField: "date",
        type: "bar"
      }},
      series: [
        {{ valueField: "target", name: "Target" }}
      ],
      legend: {{ visible: false }},
      tooltip: {{ enabled: true }},
      argumentAxis: {{
        valueMarginsEnabled: false
      }},
      // Zoom on time axis for aligned analysis
      zoomAndPan: {{
        argumentAxis: "both",
        valueAxis: "none",
        dragToZoom: true,
        allowMouseWheel: true,
        allowTouchGestures: true,
        panKey: "shift"
      }}
    }});
  </script>
</body>
</html>
"""

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html
