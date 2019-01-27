import numpy as np
import matplotlib.pyplot as plt


def plot_performance(fname, records, x_values, *,
                     offset=0,
                     capsize=7,
                     string=False):
    
    """
    Plot error bar plot of many records
    """
    num_records = len(records)
    adjusted_x_values = {}
    shift = -float(num_records) / 2.
    
    i = 0
    for label in records:
        adjusted_x_values[label] = np.array(x_values) + (shift + i) * offset
        i += 1
    
    plt.figure(figsize=(10, 7))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xticks(x_values)
    
    for label in records:
        record = records[label]
        
        if string:
            y_values = [float(record[str(x)][0]) for x in x_values]
            y_errbars = [float(record[str(x)][1]) for x in x_values]
        else:
            y_values = [float(record[x][0]) for x in x_values]
            y_errbars = [float(record[x][1]) for x in x_values]
            
        plt.errorbar(adjusted_x_values[label], y_values, yerr=y_errbars, 
                     fmt='o-', capsize=capsize, label=label)
    
    plt.legend(prop={'size': 20})
    plt.savefig(fname, transparent=True ,bbox_inches='tight')
    plt.show()