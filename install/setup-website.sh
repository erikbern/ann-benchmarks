#!/bin/sh
wget --no-check-certificate https://github.com/chartjs/Chart.js/releases/download/v2.5.0/Chart.js &&
patch < website-chart.js.2.5.0.patch &&
mkdir ../website &&
mkdir ../website/js &&
mv Chart.js ../website/js/
