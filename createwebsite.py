import matplotlib as mpl
mpl.use('Agg')
import argparse
import os, json, pickle, yaml
import numpy
import hashlib

from ann_benchmarks.main import get_fn
from ann_benchmarks import results
from ann_benchmarks import datasets
from ann_benchmarks.plotting.plot_variants import all_plot_variants as plot_variants
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils  import get_plot_label, compute_metrics, create_pointset, create_linestyles
import plot

colors = [
    "rgba(166,206,227,1)",
    "rgba(31,120,180,1)",
    "rgba(178,223,138,1)",
    "rgba(51,160,44,1)",
    "rgba(251,154,153,1)",
    "rgba(227,26,28,1)",
    "rgba(253,191,111,1)",
    "rgba(255,127,0,1)",
    "rgba(202,178,214,1)"
    ]

point_styles = {
        "o" : "circle",
        "<" : "triangle",
        "*" : "star",
        "x" : "cross",
        "+" : "rect",
        }

def convert_color(color):
    r, g, b, a = color
    return "rgba(%(r)d, %(g)d, %(b)d, %(a)d)" % {
            "r" : r * 255, "g" : g * 255,  "b" : b * 255 , "a" : a}

def convert_linestyle(ls):
    new_ls = {}
    for algo in ls.keys():
        algostyle = ls[algo]
        new_ls[algo] = (convert_color(algostyle[0]), convert_color(algostyle[1]),
                algostyle[2], point_styles[algostyle[3]])
    return new_ls

def directory_path(s):
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError("'%s' is not a directory" % s)
    return s + "/"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--plottype',
    help = 'Generate only the plots specified',
    nargs = '*',
    choices = plot_variants.keys(),
    default = plot_variants.keys())
parser.add_argument(
    '--outputdir',
    help = 'Select output directory',
    default = '.',
    type=directory_path,
    action = 'store')
parser.add_argument(
    '--definitions',
    help = 'YAML file with dataset and algorithm annotations',
    action = 'store')
parser.add_argument(
    '--limit',
    help='the maximum number of points to load from the dataset, or -1 to load all of them',
    type=int,
    default=-1)
parser.add_argument(
    '--latex',
    help='generates latex code for each plot',
    action = 'store_true')
parser.add_argument(
    '--scatter',
    help='create scatterplot for data',
    action = 'store_true')
args = parser.parse_args()

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
              <ul class="nav navbar-nav">
                <li class="active"><a href="index.html#datasets">Datasets</a></li>
              </ul>
              <ul class="nav navbar-nav">
                <li class="active"><a href="index.html#algorithms">Algorithms</a></li>
              </ul>
              <ul class="nav navbar-nav">
                <li class="active"><a href="index.html#raw">Raw Data & Configuration</a></li>
              </ul>
              <ul class="nav navbar-nav">
                <li class="active"><a href="index.html#contact">Contact</a></li>
              </ul>
            </div><!--/.nav-collapse -->
          </div>
        </nav>""" % {"title" : title}

def get_latex_plot(all_data, xm, ym, plottype):
    latex_str = """
\\begin{figure}
    \\centering
    \\begin{tikzpicture}
        \\begin{axis}[
            xlabel={%(xlabel)s},
            ylabel={%(ylabel)s},
            ymode = log,
            yticklabel style={/pgf/number format/fixed,
                              /pgf/number format/precision=3},
            legend style = { anchor=west},
            cycle list name = black white
            ]
    """ % {"xlabel" : xm["description"], "ylabel" : ym["description"] }
    color_index = 0
    only_marks = ""
    if plottype == "bubble":
        only_marks = "[only marks]"
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
            xs, ys, ls, axs, ays, als = create_pointset(algo, all_data, xn, yn)
            for i in range(len(ls)):
                if "Subprocess" in ls[i]:
                    ls[i] = ls[i].split("(")[1].split("{")[1].split("}")[0].replace("'", "")
            latex_str += """
        \\addplot %s coordinates {""" % only_marks
            for i in range(len(xs)):
                latex_str += "(%s, %s)" % (str(xs[i]), str(ys[i]))
            latex_str += " };"
            latex_str += """
        \\addlegendentry{%s};
            """ % (algo)
    latex_str += """
        \\end{axis}
    \\end{tikzpicture}
    \\caption{%(caption)s}
    \\label{}
\\end{figure}
    """ % {"caption" : get_plot_label(xm, ym)}
    return latex_str

def create_data_points(all_data, xn, yn, linestyle, render_all_points):
    color_index = 0
    html_str = ""
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
            xs, ys, ls, axs, ays, als = create_pointset(algo, all_data, xn, yn)
            if render_all_points:
                xs, ys, ls = axs, ays, als
# TODO Put this somewhere else
# pretty print subprocess parameter settings.
            for i in range(len(ls)):
                if "Subprocess" in ls[i]:
                    ls[i] = ls[i].split("(")[1].split("{")[1].split("}")[0].replace("'", "")
            html_str += """
                {
                    label: "%(algo)s",
                    fill: false,
                    pointStyle: "%(ps)s",
                    borderColor: "%(color)s",
                    data: [ """ % {"algo" : algo, "color" : linestyle[algo][0], "ps" : linestyle[algo][3] }

            for i in range(len(xs)):
                html_str += """
                        { x: %(x)s, y: %(y)s, label: "%(label)s" },""" % {"x" : str(xs[i]), "y" : str(ys[i]), "label" : ls[i] }
            html_str += """
                ]},"""
            color_index += 1
    return html_str

def create_plot(ds, all_data, xn, yn, linestyle, additional_label = "", plottype = "line"):
    xm, ym = (metrics[xn], metrics[yn])
    output_str = """
        <h3>%(xlabel)s/%(ylabel)s</h3>
        <div id="%(xlabel)s%(ylabel)s%(label)s">
            <canvas id="chart%(xlabel)s%(ylabel)s%(label)s" width="800" height="600"></canvas>
            <script>
                var ctx = document.getElementById("chart%(xlabel)s%(ylabel)s%(label)s");
                var chart = new Chart(ctx, {
                    type: "%(plottype)s",
                    data: { datasets: [%(datapoints)s
                        ]},
                        options: {
                            responsive: false,
                            title:{
                                display:true,
                                text: '%(plotlabel)s'
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
                                        labelString: ' %(ylabel)s '
                                    }
                                }]
                            }
                        }
                    });
                </script>
            </div>""" % { "id" : ds, "xlabel" :  xm["description"], "ylabel" : ym["description"], "plottype" : plottype,
                        "plotlabel" : get_plot_label(xm, ym),  "label": additional_label,
                        "datapoints" : create_data_points(all_data, xn, yn, linestyle, plottype == "bubble") }
    if args.latex:
        output_str += """
        <div class="row">
            <div class="col-md-4 text-center">
                <button type="button" id="button_%(buttonlabel)s" class="btn btn-default" >Toggle latex code</button>
            </div>
        </div>
        <script>
        $("#button_%(buttonlabel)s").click(function() {
            $("#plot_%(buttonlabel)s").toggle();
        });
        </script>
        <div id="plot_%(buttonlabel)s" style="display:none">
            <pre>
            %(latexcode)s
            </pre>
        </div>
        """ % {  "latexcode": get_latex_plot(all_data, xm, ym, plottype), "buttonlabel" : hashlib.sha224(get_plot_label(xm, ym) + additional_label).hexdigest() }
    return output_str

query_cache = {}

all_runs_by_dataset = {}
dataset_information = {}
all_runs_by_algorithm = {}

for d, r in results.get_results_with_descriptors(
        None, None, None, None, None):
    sdn = None
    if not d["query_dataset"]:
        sdn = "%(dataset)s_%(count)d_%(limit)d_%(distance)s" % d
    else:
        sdn = \
          "%(dataset)s_%(count)d_%(limit)d_%(query_dataset)s_%(distance)s" % d
    cqf = datasets.get_query_cache_path(
            d["dataset"], d["count"], d["limit"], d["distance"],
            d["query_dataset"])
    if not os.path.isfile(cqf):
        print """\
warning: query file "%s" is missing, skipping""" % cqf
        continue
    else:
        algo = d["algorithm"]
        if not cqf in query_cache:
            with open(cqf, "r") as fp:
                query_cache[cqf] = pickle.load(fp)
        ms, _ = compute_metrics(query_cache[cqf], [r])
        ms = ms[algo]

        if not algo in all_runs_by_algorithm:
            all_runs_by_algorithm[algo] = {}
        algo_ds = d["dataset"] + " (k = " + str(d["count"]) + ")"
        if not algo_ds in all_runs_by_algorithm[algo]:
            all_runs_by_algorithm[algo][algo_ds] = []
        all_runs_by_algorithm[algo][algo_ds].extend(ms)

        if not sdn in all_runs_by_dataset:
            all_runs_by_dataset[sdn] = {}
            dataset_information[sdn] = d
        if not algo in all_runs_by_dataset[sdn]:
            all_runs_by_dataset[sdn][algo] = []
        all_runs_by_dataset[sdn][algo].extend(ms)
del query_cache

# Build a website for each dataset
for (ds, runs) in all_runs_by_dataset.items():
    all_algos = runs.keys()
    linestyles = convert_linestyle(create_linestyles(all_algos))
    output_str = get_html_header(ds)
    ds_name = dataset_information[ds]["dataset"] + " (k = " + str(dataset_information[ds]["count"]) + ")"
    output_str += """
        <div class="container">
        <h2>Plots for %(id)s</h2>""" % { "id" : ds_name }
    for plottype in args.plottype:
        xn, yn = plot_variants[plottype]
        print "Processing '%s' with %s" % (ds, plottype)
        output_str += create_plot(ds_name, runs, xn, yn, linestyles)
    if args.scatter:
        output_str += """
        <hr />
        <h2>Scatterplots for %(id)s""" % { "id" : ds_name }
        for plottype in args.plottype:
            xn, yn = plot_variants[plottype]
            print "Processing scatterplot '%s' with %s" % (ds, plottype)
            output_str += create_plot(ds, runs, xn, yn, linestyles, "Scatterplot ", "bubble")
    # create png plot for summary page
    plot.create_plot(runs, True, False,
            False, True, 'k-nn', 'qps',  args.outputdir + ds + ".png",
            create_linestyles(all_algos))
    output_str += """
        </div>
    </div>
    </body>
</html>
"""
    with open(args.outputdir + ds + ".html", "w") as text_file:
        text_file.write(output_str)

# Build a website for each algorithm
# Build website. TODO Refactor with dataset code.
for (algo, runs) in all_runs_by_algorithm.items():
    all_data = runs.keys()
    output_str = get_html_header(algo)
    output_str += """
        <div class="container">
        <h2>Plots for %(id)s""" % { "id" : algo }
    for plottype in args.plottype:
        xn, yn = plot_variants[plottype]
        linestyles = convert_linestyle(create_linestyles(all_data))
        print "Processing '%s' with %s" % (algo, plottype)
        output_str += create_plot(algo, runs, xn, yn, linestyles)
    plot.create_plot(runs, True, False,
            False, True, 'k-nn', 'qps',  args.outputdir + algo + ".png",
            create_linestyles(all_data))
    output_str += """
    </div>
    </body>
</html>
"""
    with open(args.outputdir + algo + ".html", "w") as text_file:
        text_file.write(output_str)

# Build an index page
with open(args.outputdir + "index.html", "w") as text_file:
    try:
        with open(args.definitions) as f:
            definitions = yaml.load(f)
    except:
        print "Could not load definitions file, annotations not available."
        definitions = {}
    output_str = get_html_header("ANN-Benchmarks")
    output_str += """
        <div class="container">
            <h1>Info</h1>
            <p>ANN-Benchmarks is a benchmarking environment for approximate nearest neighbor algorithms search. This website contains the current benchmarking results. Please visit <a href="http://github.com/maumueller/ann-benchmarks/">http://github.com/maumueller/ann-benchmarks/</a> to get an overview over evaluated data sets and algorithms. Make a pull request on <a href="http://github.com/maumueller/ann-benchmarks/">Github</a> to add your own code or improvements to the
            benchmarking system.
            </p>
            <div id="results">
            <h1>Benchmarking Results</h1>
            <p>Results are split by distance measure and dataset. In the bottom, you can find an overview of an algorithm's performance on all datasets. Each dataset is annoted
            by <em>(k = ...)</em>, the number of nearest neighbors an algorithm was supposed to return. The plot shown depicts <em>Recall</em> (the fraction
            of true nearest neighbors found, on average over all queries) against <em>Queries per second</em>.  Clicking on a plot reveils detailled interactive plots, including
            approximate recall, index size, and build time.</p>
            <h2 id ="datasets">Results by Dataset</h2>
            """

    distance_measures = sorted(set([e.split("_")[-1] for e in all_runs_by_dataset.keys()]))
    datasets = sorted(set([e.split("_")[0] for e in all_runs_by_dataset.keys()]))
    for dm in distance_measures:
        output_str += """
            <h3>Distance: %s</h3>
            """ % dm.capitalize()
        for ds in datasets:
            for idd in sorted([e for e in all_runs_by_dataset.keys() \
                    if e.split("_")[0] == ds and e.split("_")[-1] == dm], \
                    key = lambda elem: int(elem.split("_")[1])):
                ds_name = dataset_information[idd]["dataset"] + " (k = " + \
                    str(dataset_information[idd]["count"]) + ")"
                output_str += """
            <a href="./%(id)s.html">
            <div class="row" id="%(id)s">
                <div class = "col-md-4 bg-success">
                    <h4>%(name)s</h4>
                    <dl class="dl-horizontal">
                    """ % { "name" : ds_name, "id" : idd }
#        if "datasets" in definitions and ds in definitions["datasets"]:
#            for k in definitions["datasets"][ds]:
#                output_str += """
#                        <dt>%s</dt>
#                        <dd>%s</dd>
#                        """ % (k, str(definitions["datasets"][ds][k]))
                output_str += """
                        </dl>
                    </div>
                    <div class = "col-md-8">
                        <img class = "img-responsive" src="%(name)s.png" />
                    </div>
                </div>
                </a>""" % { "name" : idd }
        output_str += """
            <hr />
        """
    output_str += """
        <h2 id="algorithms">Results by Algorithm</h2>
        <ul class="list-inline"><b>Algorithms:</b>"""
    algorithms = all_runs_by_algorithm.keys()
    for algo in algorithms:
        output_str += '<li><a href="#%(name)s">%(name)s</a></li>' % {"name" : algo}
    output_str += "</ul>"
    for algo in algorithms:
        output_str += """
            <a href="./%(name)s.html">
            <div class="row" id="%(name)s">
                <div class = "col-md-4 bg-success">
                    <h4>%(name)s</h4>
                    <dl class="dl-horizontal">
                    """ % { "name" : algo }
        if "alogs" in definitions and algo in definitions["algos"]:
            for k in definitions["algos"][algo]:
                output_str += """
                        <dt>%s</dt>
                        <dd>%s</dd>
                        """ % (k, str(definitions["algos"][algo][k]))
        output_str += """
                </dl>
            </div>
            <div class = "col-md-8">
                <img class = "img-responsive" src="%(name)s.png" />
            </div>
        </div>
        </a>
        <hr />""" % { "name" : algo}
    output_str += """
            <h2 id ="raw">Raw Data & Configuration</h2>
            <p>Please find the raw experimental data <a href="./results-sisap.tar">here</a> (13 GB). The query set is available <a href="./queries-sisap.tar">queries-sisap.tar</a> (8 GB) as well. The algorithms used the following parameter choices in the experiments: <a href="./algos.yaml">k = 10</a> and <a href="./algos100.yaml">k=100</a>.</p>
            <div id="contact">
            <h2>Contact</h2>
            <p>ANN-Benchmarks has been developed by Martin Aumueller (maau@itu.dk), Erik Bernhardsson (mail@erikbern.com), and Alec Faitfull (alef@itu.dk). Please use
            <a href="https://github.com/maau/ann-benchmarks/">Github</a> to submit your implementation or improvements.</p>
            </div>
        </div>
    </body>
</html>"""
    text_file.write(output_str)


