from Bio import Phylo
import matplotlib
import matplotlib.pyplot as plt

tree = Phylo.read("data/archaeopteryx_2009_Guam.xml", "phyloxml")
tree.rooted = True
str_tree = str(tree)
seq_list = list()
seq_dict = dict()
for s in str_tree.split("name="):
    if "GenBank" in s:
        seq_list.append(s)


def get_seq(sub_seq):
    s = sub_seq.split("\n")
    ID = s[0][1:-2]
    genbank_id = s[2].strip()
    seq = s[3].strip().replace("-", "")
    return ID, genbank_id, seq


for sub_seq in seq_list:
    ID, genbank_id, seq = get_seq(sub_seq)
    seq_dict[ID] = (genbank_id, seq)

print(seq_dict)

matplotlib.rc('font', size=18)
# set the size of the figure
fig = plt.figure(figsize=(30, 20), dpi=40)
# alternatively
# fig.set_size_inches(10, 20)
axes = fig.add_subplot(1, 1, 1)
tree.root.color = "red"
# Phylo.draw(tree, axes=axes)

terms = tree.get_terminals()
dist_dict = dict()
for t in terms:
    dist_dict[str(t)] = tree.distance(t)

sort_dist = sorted(dist_dict.items(), key=lambda x: x[1])
print(sort_dist)
sort_name = [k for k, v in sort_dist]

raw_data = str()
for k in sort_name:
    genbank_id, seq = seq_dict[k]
    raw_data += seq
    raw_data += "\n"

file = open("data/archaeopteryx_2009_Guam.txt", "w")
file.write(raw_data)

file.close()
