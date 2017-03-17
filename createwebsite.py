import argparse
import os, json, pickle
import numpy
from collections import deque

from ann_benchmarks.main import get_fn
from ann_benchmarks.plotting.metrics import all_metrics as metrics

colors = deque(["rgba(151,187,205,1)", "rgba(220,220,220,1)", "rgba(247,70,74,1)", "rgba(70,191,189,1)",
          "rgba(253,180,92,1)", "rgba(148,159,177,1)", "rgba(77,83,96,1)"])

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    nargs = '*')
parser.add_argument(
    '--precision',
    help = 'Which precision metric to use',
    nargs = '*',
    choices = metrics.keys(),
    default = [metrics.keys()[0]],
    )
parser.add_argument(
    '--outputdir',
    help = 'Select output directory',
    action = 'store'
    )
args = parser.parse_args()

outputdir = ""

if args.outputdir != None:
    outputdir = args.outputdir

def get_html_header(title):
    return """
<!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
        <title>%(title)s</title>
        <script src="js/Chart.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
        <!-- Include all compiled plugins (below), or include individual files as needed -->
        <script src="js/bootstrap.min.js"></script>
        <!-- Bootstrap -->
        <link href="css/bootstrap.min.css" rel="stylesheet">
        <link href="css/style.css" rel="stylesheet">

        <!-- HTML5 shim and Respond.js for IE8 support of HTML5 elements and media queries -->
        <!-- WARNING: Respond.js doesn't work if you view the page via file:// -->
        <!--[if lt IE 9]>
          <script src="https://oss.maxcdn.com/html5shiv/3.7.3/html5shiv.min.js"></script>
          <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
        <![endif]-->
      </head>
         <body>

        <nav class="navbar navbar-inverse navbar-fixed-top">
          <div class="container">
            <div class="navbar-header">
              <a class="navbar-brand" href="index.html">ANN Benchmarks</a>
            </div>
            <div id="navbar" class="collapse navbar-collapse">
              <ul class="nav navbar-nav">
                <li class="active"><a href="index.html">Home</a></li>
              </ul>
            </div><!--/.nav-collapse -->
          </div>
        </nav>""" % {"title" : title}

def create_plot(ds, all_data, metric):
    output_str = """
        <h2>%(id)s with %(metric)s</h2>
        <canvas id="chart%(metric)s" width="800" height="600"></canvas>
        <script>
            var ctx = document.getElementById("chart%(metric)s");
            var chart = new Chart(ctx, {
                type: "line",
                data: { datasets: [""" % { "id" : ds, "metric" :  metric["description"] }
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
            data = all_data[algo]
            data.sort(key=lambda t: t[-2]) # sort by time
            ys = [1.0 / t[-2] for t in data] # queries per second
            xs = [t[-1] for t in data]
            ls = [t[1] for t in data]
# TODO Put this somewhere else
# pretty print subprocess parameter settings.
            for i in range(len(ls)):
                if "Subprocess" in ls[i]:
                    ls[i] = ls[i].split("(")[1].split("{")[1].split("}")[0]
            # Plot Pareto frontier
            xs, ys, sl = [], [], []
            last_y = metric["initial-y"]
            for t in data:
                y = t[-1]
                if metric["plot"](y, last_y):
                    last_y = y
                    xs.append(t[-1])
                    ys.append(1.0 / t[-2])
		    ls.append(t[1])
            output_str += """
                {
                    label: "%(algo)s",
                    fill: false,
                    borderColor: "%(color)s",
                    data: [ """ % {"algo" : algo, "color" : colors[0] }

            for i in range(len(xs)):
                output_str += """
                        { x: %(x)s, y: %(y)s, label: "%(label)s" },""" % {"x" : str(xs[i]), "y" : str(ys[i]), "label" : ls[i] }
            output_str += """
                ]},"""
            colors.rotate(1)

    output_str += """
            ]}, options: {
                        responsive: false,
                        title:{
                            display:true,
                            text:'Precision-Performance tradeoff - up and to the right is better'
                        },
                        scales: {
                            xAxes: [{
                                display: true,
                                type: 'linear',
                                max: '1',
                                position: 'bottom',
                                scaleLabel: {
                                    display: true,
                                    labelString: ' %(xlabel)s   '
                                }
                            }],
                            yAxes: [{
                                display: true,
                                type: 'logarithmic',
                                scaleLabel: {
                                    display: true,
                                    labelString: 'Queries per second - larger is better'
                                }
                            }]
                        }
                    }
                }); """ % { "xlabel" :  metric["description"]}

    output_str += """
        </script>
        """
    return output_str

all_data = {} # all_data[metric][algo] = []
for ds in args.dataset:
    results_fn = get_fn("results", ds)
    queries_fn = get_fn("queries", ds)
    assert os.path.exists(queries_fn), """\
the queries file '%s' is missing""" % queries_fn

    queries = pickle.load(open(queries_fn))
    with open(get_fn("results", ds)) as f:
        for line in f:
            run = json.loads(line)
            algo = run["library"]
            algo_name = run["name"]
            build_time = run["build_time"]
            search_time = run["best_search_time"]

            print "--"
            print algo_name
            for metric_name in args.precision:
                metric = metrics[metric_name]
                precision = metric["function"](queries, run)
                print "%s: %g" % (metric_name, precision)
                # Should build_time and search_time really go in here?
                all_data.setdefault(
                    metric_name, {}).setdefault(
                    algo, []).append((algo,
                            algo_name,
                            float(build_time),
                            float(search_time),
                            precision))

# Build a website for each dataset
for ds in args.dataset:
    output_str = get_html_header(ds)
    output_str += """
        <div class="container">
        <h2>Plots for %(id)s""" % { "id" : ds }
    for metric_name in args.precision:
        metric = metrics[metric_name]
        print "Processing '%s' with %s" % (ds, metric["description"])
        output_str += create_plot(ds, all_data[metric_name], metric)

    output_str += """
    </div>
    </body>
</html>
"""
    with open(outputdir + ds + ".html", "w") as text_file:
        text_file.write(output_str)

# Build an index page
with open(outputdir + "index.html", "w") as text_file:
    output_str = get_html_header("ANN-Benchmarks")
    output_str += """
        <div class="container">
        <h2>Overview over Datasets</h2>
        <p>Click on a dataset to see the performance/quality plots.</p>
        <ul>"""
    for ds in args.dataset:
        output_str += """
            <li><a href="%(id)s.html">%(id)s</a></li>""" % { "id" : ds }
    output_str += """
        </ul>
        </div>
    </body>
</html>"""
    text_file.write(output_str)

