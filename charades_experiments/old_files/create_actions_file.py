import os
import sys
import csv

with open('Charades_v1_classes.txt') as f_in:
    out_name = 'Charades_v1_actions.csv'
    if os.path.exists(out_name):
        os.remove(out_name)
    f_out = open(out_name, 'w+')
    writer = csv.writer(f_out, delimiter=',')
    for line in f_in:
        line = line.strip('\n').split()[0]
        writer.writerow([line])
