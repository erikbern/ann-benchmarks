import matplotlib as mpl
mpl.use('Agg')
import argparse
import os, json, pickle, yaml
import numpy
import hashlib

from ann_benchmarks import results
from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.plot_variants import all_plot_variants as plot_variants
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils  import get_plot_label, compute_metrics, compute_all_metrics, create_pointset, create_linestyles
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

def get_run_desc(properties):
    return "%(dataset)s_%(count)d_%(distance)s" % properties

def get_dataset_from_desc(desc):
    return desc.split("_")[0]

def get_count_from_desc(desc):
    return desc.split("_")[1]

def get_distance_from_desc(desc):
    return desc.split("_")[2]

def get_dataset_label(desc):
    return get_dataset_from_desc(desc) + " (k = " + get_count_from_desc(desc) + ")"

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
        <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.5.0/Chart.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
        <!-- Include all compiled plugins (below), or include individual files as needed -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
        <!-- Bootstrap -->
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
        <style>
            body { padding-top: 50px; }
        </style>
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
                <li class="active"><a href="index.html#contact">Contact</a></li>
              </ul>
            </div><!--/.nav-collapse -->
          </div>
        </nav>""" % {"title" : title}

def get_index_description():
    return """
        <div class="container">
            <h1>Info</h1>
            <p>ANN-Benchmarks is a benchmarking environment for approximate nearest neighbor algorithms search. This website contains the current benchmarking results. Please visit <a href="http://github.com/erikbern/ann-benchmarks/">http://github.com/erikbern/ann-benchmarks/</a> to get an overview over evaluated data sets and algorithms. Make a pull request on <a href="http://github.com/erikbern/ann-benchmarks/">Github</a> to add your own code or improvements to the
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

def get_index_footer():
    return """
            <div id="contact">
            <h2>Contact</h2>
            <p>ANN-Benchmarks has been developed by Martin Aumueller (maau@itu.dk), Erik Bernhardsson (mail@erikbern.com), and Alec Faitfull (alef@itu.dk). Please use
            <a href="https://github.com/erikbern/ann-benchmarks/">Github</a> to submit your implementation or improvements.</p>
            </div>
        </div>
    </body>
</html>"""

def get_row_desc(idd, desc):
    return """
        <a href="./%(idd)s.html">
            <div class="row" id="%(idd)s">
                <div class = "col-md-4 bg-success">
                    <h4>%(desc)s</h4>
            </div>
            <div class = "col-md-8">
                <img class = "img-responsive" src="%(idd)s.png" />
            </div>
        </div>
        </a>
        <hr />""" % { "idd" : idd, "desc" : desc}

def prepare_data(data, xn, yn):
    """Change format from (algo, instance, dict) to (algo, instance, x, y)."""
    res = []
    for algo, algo_name, result in data:
        res.append((algo, algo_name, result[xn], result[yn]))
    return res

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
            xs, ys, ls, axs, ays, als = create_pointset(prepare_data(all_data[algo], xn, yn), xn, yn)
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
            xs, ys, ls, axs, ays, als = create_pointset(prepare_data(all_data[algo], xn, yn), xn, yn)
            if render_all_points:
                xs, ys, ls = axs, ays, als
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
                function pushOrConcat(base, toPush) {
                        if (toPush) {
                                if (Chart.helpers.isArray(toPush)) {
                                        // base = base.concat(toPush);
                                        Array.prototype.push.apply(base, toPush);
                                } else {
                                        base.push(toPush);
                                }
                        }

                        return base;
                }
                Chart.Tooltip.prototype.getFooter = function(tooltipItem, data) {
                    var me = this;
                    var callbacks = me._options.callbacks;
                    var item = tooltipItem[0];

                    var beforeFooter = callbacks.beforeFooter.apply(me, arguments);
                    var footer = "Parameters: " + data.datasets[item.datasetIndex].data[item.index].label || '';
                    var afterFooter = callbacks.afterFooter.apply(me, arguments);

                    var lines = [];
                    lines = pushOrConcat(lines, beforeFooter);
                    lines = pushOrConcat(lines, footer);
                    lines = pushOrConcat(lines, afterFooter);

                    return lines;
                }

                </script>
            </div>""" % { "id" : ds, "xlabel" :  xm["description"], "ylabel" : ym["description"], "plottype" : plottype,
                        "plotlabel" : get_plot_label(xm, ym),  "label": additional_label,
                        "datapoints" : create_data_points(all_data,
                            xn, yn, linestyle, plottype == "bubble") }
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

def build_detail_site(data, label_func):
    for (name, runs) in data.items():
        all_runs = runs.keys()
        linestyles = convert_linestyle(create_linestyles(all_runs))
        output_str = get_html_header(name)
        label = label_func(name)
        output_str += """
            <div class="container">
            <h2>Plots for %s</h2>""" % (label)
        for plottype in args.plottype:
            xn, yn = plot_variants[plottype]
            print("Processing '%s' with %s" % (name, plottype))
            output_str += create_plot(label, runs, xn, yn, linestyles)
        if args.scatter:
            output_str += """
            <hr />
            <h2>Scatterplots for %s""" % (label)
            for plottype in args.plottype:
                xn, yn = plot_variants[plottype]
                print("Processing scatterplot '%s' with %s" % (name, plottype))
                output_str += create_plot(name, runs, xn, yn, linestyles, "Scatterplot ", "bubble")
        # create png plot for summary page
        data_for_plot = {}
        for k in runs.keys():
            data_for_plot[k] = prepare_data(runs[k], 'k-nn', 'qps')
        plot.create_plot(data_for_plot, False,
                False, True, 'k-nn', 'qps',  args.outputdir + name + ".png",
                create_linestyles(all_runs))
        output_str += """
            </div>
        </div>
        </body>
    </html>
    """
        with open(args.outputdir + name + ".html", "w") as text_file:
            text_file.write(output_str)


def build_index(datasets, algorithms):
    distance_measures = sorted(set([get_distance_from_desc(e) for e in datasets.keys()]))
    sorted_datasets = sorted(set([get_dataset_from_desc(e) for e in datasets.keys()]))

    output_str = get_html_header("ANN-Benchmarks")
    output_str += get_index_description()

    for dm in distance_measures:
        output_str += """
            <h3>Distance: %s</h3>
            """ % dm.capitalize()
        for ds in sorted_datasets:
            matching_datasets = [e for e in datasets.keys() \
                    if get_dataset_from_desc(e) == ds and \
                       get_distance_from_desc(e) == dm]
            sorted_matches = sorted(matching_datasets, \
                    key = lambda e: int(get_count_from_desc(e)))
            for idd in sorted_matches:
                output_str += get_row_desc(idd, get_dataset_label(idd))
    output_str += """
        <h2 id="algorithms">Results by Algorithm</h2>
        <ul class="list-inline"><b>Algorithms:</b>"""
    algorithm_names = algorithms.keys()
    for algo in algorithm_names:
        output_str += '<li><a href="#%(name)s">%(name)s</a></li>' % {"name" : algo}
    output_str += "</ul>"
    for algo in algorithm_names:
        output_str += get_row_desc(algo, algo)
    output_str += get_index_footer()

    with open(args.outputdir + "index.html", "w") as text_file:
        text_file.write(output_str)

def load_all_results():
    """Read all result files and compute all metrics"""
    all_runs_by_dataset = {}
    all_runs_by_algorithm = {}
    for f in results.load_all_results():
        properties = dict(f.attrs)
        # TODO Fix this properly. Sometimes the hdf5 file returns bytes
        # This converts these bytes to strings before we work with them
        for k in properties.keys():
            try:
                properties[k]= properties[k].decode()
            except:
                pass
        sdn = get_run_desc(properties)
        dataset = get_dataset(properties["dataset"])
        algo = properties["algo"]
        ms = compute_all_metrics(dataset, f, properties["count"], properties["algo"])
        algo_ds = get_dataset_label(sdn)

        all_runs_by_algorithm.setdefault(algo, {}).setdefault(algo_ds, []).append(ms)
        all_runs_by_dataset.setdefault(sdn, {}).setdefault(algo, []).append(ms)
    return (all_runs_by_dataset, all_runs_by_algorithm)

runs_by_ds, runs_by_algo = load_all_results()
build_detail_site(runs_by_ds, lambda label: get_dataset_label(label))
build_detail_site(runs_by_algo, lambda x: x)
build_index(runs_by_ds, runs_by_algo)
