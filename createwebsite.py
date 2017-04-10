import argparse
import os, json, pickle
import numpy

from ann_benchmarks.main import get_fn
from ann_benchmarks.plotting.plot_variants import all_plot_variants as plot_variants
from ann_benchmarks.plotting.utils  import get_plot_label, create_pointset, load_results, create_linestyles

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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    nargs = '*')
parser.add_argument(
    '--plottype',
    help = 'Which plots to generate',
    nargs = '*',
    choices = plot_variants.keys(),
    default = [plot_variants.keys()[0]],
    )
parser.add_argument(
    '--outputdir',
    help = 'Select output directory',
    action = 'store'
    )
parser.add_argument(
    '--limit',
    help='the maximum number of points to load from the dataset, or -1 to load all of them',
    type=int,
    default=-1)
parser.add_argument(
    '--latex',
    help='generates latex code for each plot',
    action = 'store_true')
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
              <ul class="nav navbar-nav">
                <li class="active"><a href="index.html#results">Results</a></li>
              </ul>
              <ul class="nav navbar-nav">
                <li class="active"><a href="index.html#contact">Contact</a></li>
              </ul>
            </div><!--/.nav-collapse -->
          </div>
        </nav>""" % {"title" : title}

def create_plot(ds, all_data, xm, ym, linestyle):
    output_str = """
        <h2>%(id)s with %(xmetric)s/%(ymetric)s</h2>
        <canvas id="chart%(xmetric)s%(ymetric)s" width="800" height="600"></canvas>
        <script>
            var ctx = document.getElementById("chart%(xmetric)s%(ymetric)s");
            var chart = new Chart(ctx, {
                type: "line",
                data: { datasets: [""" % { "id" : ds, "xmetric" :  xm["description"], "ymetric" : ym["description"] }
    color_index = 0
    for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
            xs, ys, axs, ays, ls = create_pointset(algo, all_data, xm, ym)
# TODO Put this somewhere else
# pretty print subprocess parameter settings.
            for i in range(len(ls)):
                if "Subprocess" in ls[i]:
                    ls[i] = ls[i].split("(")[1].split("{")[1].split("}")[0].replace("'", "")
            output_str += """
                {
                    label: "%(algo)s",
                    fill: false,
                    pointStyle: "%(ps)s",
                    borderColor: "%(color)s",
                    data: [ """ % {"algo" : algo, "color" : linestyle[algo][0], "ps" : linestyle[algo][3] }

            for i in range(len(xs)):
                output_str += """
                        { x: %(x)s, y: %(y)s, label: "%(label)s" },""" % {"x" : str(xs[i]), "y" : str(ys[i]), "label" : ls[i] }
            output_str += """
                ]},"""
            color_index += 1

    output_str += """
            ]}, options: {
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
                }); """ % { "xlabel" :  xm["description"], "ylabel" : ym["description"],
                        "plotlabel" : get_plot_label(xm, ym)}

    output_str += """
        </script>
        """
    if args.latex:
        output_str += """
        <h3>Latex code for plot</h3>
        <pre>
\\begin{figure}
    \\centering
    \\begin{tikzpicture}
        \\begin{axis}[
            xlabel={%(xlabel)s},
            ylabel={%(ylabel)s},
            yticklabel style={/pgf/number format/fixed,
                              /pgf/number format/precision=3},
            legend style = { anchor=west},
            cycle list name = black white
            ]
        """ % {"xlabel" : xm["description"], "ylabel" : ym["description"]}
        color_index = 0
        for algo in sorted(all_data.keys(), key=lambda x: x.lower()):
                xs, ys, axs, ays, ls = create_pointset(algo, all_data, xm, ym)
                for i in range(len(ls)):
                    if "Subprocess" in ls[i]:
                        ls[i] = ls[i].split("(")[1].split("{")[1].split("}")[0].replace("'", "")
                output_str += """
            \\addplot coordinates {"""
                for i in range(len(xs)):
                    output_str += "(%s, %s)" % (str(xs[i]), str(ys[i]))
                output_str += " };"
                output_str += """
            \\addlegendentry{%s};
                """ % (algo)
        output_str += """
        \\end{axis}
    \\end{tikzpicture}
    \\caption{%(caption)s}
    \\label{}
\\end{figure}
</pre>
        """ % { "caption" : get_plot_label(xm, ym) }
    return output_str


# Build a website for each dataset
for ds in args.dataset:
    output_str = get_html_header(ds)
    output_str += """
        <h2>Plots for %(id)s""" % { "id" : ds }
    for plottype in args.plottype:
        xm, ym = plot_variants[plottype]
        runs, all_algos = load_results(args.dataset, xm, ym, args.limit)
        linestyles = convert_linestyle(create_linestyles(all_algos))
        print "Processing '%s' with %s" % (ds, plottype)
        output_str += create_plot(ds, runs[ds], xm, ym, linestyles)

    output_str += """
    </div>
    </body>
</html>
"""
    with open(outputdir + ds + ".html", "w") as text_file:
        text_file.write(output_str)

# Build a website for each algorithm
# Get all algorithms
xm, ym = plot_variants[args.plottype[0]]
_, all_algos = load_results(args.dataset, xm, ym, args.limit)
# Build website. TODO Refactor with dataset code.
for algo in all_algos:
    output_str = get_html_header(algo)
    output_str += """
        <div class="container">
        <h2>Plots for %(id)s""" % { "id" : algo }
    for plottype in args.plottype:
        xm, ym = plot_variants[plottype]
        runs, all_algos = load_results(args.dataset, xm, ym, args.limit)
        all_data = {}
        for ds in runs:
            all_data[ds] = runs[ds][algo]
        linestyles = convert_linestyle(create_linestyles(args.dataset))
        print "Processing '%s' with %s" % (algo, plottype)
        output_str += create_plot(algo, all_data, xm, ym, linestyles)
    output_str += """
    </div>
    </body>
</html>
"""
    with open(outputdir + algo + ".html", "w") as text_file:
        text_file.write(output_str)

# Build an index page
with open(outputdir + "index.html", "w") as text_file:
    output_str = get_html_header("ANN-Benchmarks")
    output_str += """
        <div class="container">
            <h2>Info</h2>
            <p>ANN-Benchmarks is a benchmarking environment for approximate nearest neighbor algorithms.</p>
            <h2>Benchmarking Results</h2>
            Results are split by dataset and by algorithm. Click on the plot to get details.
            <div class="results">
            <h3>... by dataset</h3>
        """
    for ds in args.dataset:
        output_str += """
            <div class="row">
                <div class = "col-md-4">
                    <h4>%(name)s</h4>
                    <dl class="dl-horizontal">
                        <dt>points</dt>
                        <dd>%(points)s</dd>
                        <dt>metric</dt>
                        <dd>%(metric)s</dd>
                        <dt>dimensions</dt>
                        <dd>%(dimension)s</dd>
                    </dl>
                </div>
                <div class = "col-md-8">
                    <img class = "img-responsive" src="%(name)s.png" />
                </div>
            </div>""" % { "name" : ds, "points" : "", "metric" : "", "dimension" : "" }
    output_str += """
        <h3>... by algorithm</h3>
        """
    for algo in all_algos:
        output_str += """
            <div class="row">
                <div class = "col-md-4">
                    <h4>%(name)s</h4>
                </div>
                <div class = "col-md-8">
                    <img class = "img-responsive" src="%(name)s.png" />
                </div>
            </div>""" % { "name" : algo}
    output_str += """
            <div class="contact">
            <h2>Contact</h2>
            </div>
        </div>
    </body>
</html>"""
    text_file.write(output_str)


