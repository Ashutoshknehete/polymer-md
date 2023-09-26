def format_plots(plt):

    #overall font issues
    plt.rcParams["font.size"] = 16                   #default font size
    plt.rcParams["axes.labelsize"] = 20             #axis labels
    plt.rcParams["axes.titlesize"] = 20             #if there is a title
    plt.rcParams["font.sans-serif"] = "Arial"       #make font Arial (except math)
    plt.rcParams["figure.autolayout"] = True        #makes tight plot
    plt.rcParams['text.usetex'] = False             #use LaTeX
    plt.rcParams["savefig.dpi"] = 600               #dpi for figures

    #making nicer tick marks
    plt.rcParams["xtick.top"] = True                #put x ticks on top
    plt.rcParams["ytick.right"] = True              #put y ticks on top
    plt.rcParams["xtick.minor.visible"] = True      #add minor x ticks
    plt.rcParams["ytick.minor.visible"] = True      #add minor y ticks
    plt.rcParams["xtick.labelsize"] = 16            #tick mark font size
    plt.rcParams["ytick.labelsize"] = 16            #tick mark font size
    plt.rcParams["xtick.major.size"] = 6.5          #major tick
    plt.rcParams["ytick.major.size"] = 6.5
    plt.rcParams["xtick.minor.size"] = 3.5
    plt.rcParams["ytick.minor.size"] = 3.5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    plt.rcParams["legend.fontsize"] = 16            #legend font
    plt.rcParams["legend.frameon"] = False          #remove box


    plt.rcParams["errorbar.capsize"] = 10
    plt.rcParams["lines.markersize"] = 7
    plt.rcParams["lines.linewidth"] = 1.2


    #set some colors
    colors = {
        "black": "#000000",
        "gray" : "#D3D3D3",
        "darkgray": "#ABB0B8",
        "charcoal": "#36454F",
        "cornsilk": "#FFF8DC",
        "almond": "#FFEBCD"
        }

    return colors
