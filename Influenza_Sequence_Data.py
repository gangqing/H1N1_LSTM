#!/usr/bin/env python
# coding: utf-8

# In[1]:

# sys.path.append("..")


# In[2]:


# we show the first 3 only
# for i, seq_record in enumerate(SeqIO.parse("663130668554-GeneFastaResults.fasta", "fasta")):
#     print(seq_record.id)
#     print(type(seq_record.id))
#     print(repr(seq_record.seq))
#     print(len(seq_record))
#     if i == 2:
#         break

# ex_seq = SeqIO.parse(
#     "toy_data/GeneFastaResults_Australia_5_8_2020.fasta", "fasta")
# print("Total number of genes: {}".format(len(list(ex_seq))))

# for i, seq_record in enumerate(ex_seq):
#     print("T: ", seq_record.description)

# In[3]:


from Bio import Phylo
import matplotlib
import matplotlib.pyplot as plt


# In[6]:


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

# Phylo.draw_ascii(tree)

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


# new_tree = Phylo.read("data/blast_tree.nwk", "newick")
# Phylo.draw_ascii(new_tree)

# In[5]:


import networkx
import pylab

# net = Phylo.to_networkx(tree)

# networkx.draw(net, with_labels=True, font_size=20)
# pylab.show()


# for n in iter(net):
#     if len(list(net.predecessors(n))) == 0:
#         root_list = list(net.successors(n))
#
# for r in root_list:
#     if str(r) == 'Clade':
#         root_list.remove(r)
#
# first_root = root_list[0]
# print(first_root)
#
# print(list(net.successors(first_root)))
