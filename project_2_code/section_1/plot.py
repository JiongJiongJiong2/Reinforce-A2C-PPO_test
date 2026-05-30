import matplotlib.pyplot as plt
import numpy as np
import os


def read_log(path, metric):
    run_log = []
    unified_logs = []
    min_lenght = np.inf
    for filename in os.listdir(path):
        if metric in filename and ".txt" in filename:
            file_log = []
            with open(path + filename, "r") as f:
                for line in f:
                    ret = float(line.strip())
                    file_log.append(ret)
                if len(file_log)< min_lenght:
                    min_lenght = len(file_log)
            run_log.append(file_log)
    for log in run_log:
        unified_logs.append(log[:min_lenght])
    return unified_logs



def plot_log(name, log, color, shadow):
    '''
    :param name: name to be used in the legend of figure
    :param log: data to be plotted
    :param color:
    :param shadow: "SE" standard error, "STD" standard deviation
    :return:
    '''
    mean = np.mean(log, axis=0)
    std = np.std(log, axis=0)
    plt.plot(mean, color, label=str(name))
    if shadow == "SE":
        std_error = std / (len(log) ** 0.5)
    elif shadow == "STD":
        std_error = std
    plt.fill_between(range(len(log[0])), mean - std_error, mean + std_error, color=color, alpha=0.3)

def plot(data, metric ,title):
    '''
    :param data: list of tuples (path, legend) of directories containing the data you want to plot in one figure
                if there are more than one file in each path containing the name "metric" it plots average of them
                with shadow showing standard error (you can change shadow to "STD" to plot standard deviation)
    :param metric: a keyword in the name of the files in each directory containing desired data to be plotted
    :param title: title of the final figure
    :return:
    '''
    for i in range(len(data)):
        path, name = data[i]
        logs = read_log(path,metric)
        plot_log(name, logs, color="C" + str(i), shadow="SE")

    plt.xlabel('Episode')
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(path + metric+'.png')
    plt.close()


# To plot average keyword (eg. return) of 3 runs of both experiment1 and experiment2 in a same figure,
#  you should have the file structure as follows:

# ./data
#    /experiment1_path/
#       /keyword0.txt
#       /keyword1.txt
#       /keyword2.txt
#    /experiment2_path/
#       /keyword0.txt
#       /keyword1.txt
#       /keyword2.txt

# Then use:
# plot([("./data/experiment1_path/","experiment1"),("./data/experiment2_path/","experiment2")],"keyword","title")

# The result will be a file named "keyword.png" in ./data/experiment2_path/".
