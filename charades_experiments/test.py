import os
import csv

# def read_csv_input(self, csv_path, data_root):
#     csv_data = []
#     with open(csv_path) as csvfile:
#         csv_reader = csv.reader(csvfile, delimiter=';')
#         for row in csv_reader:
#             # 
#             item = ListDataJpeg(row[0],
#                                 row[1],
#                                 row[2], # list of actions
#                                 row[3], # scene
#                                 os.path.join(data_root, row[0])
#                                 )
#             csv_data.append(item)
#     return csv_data
with open('data/Charades_v1_train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    i = 0
    for row in reader:

        l = [b for a in row['actions'].split(';') for b in a.split() if 'c' in b]
        # for a in row['actions'].split(';'):
        #     for b in a.split():
        #         if 'c' in b:
        #             print(b)
        print(l)
        print()
        i += 1
        if i == 5: break
        