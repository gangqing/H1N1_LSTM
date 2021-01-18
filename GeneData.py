

class Data:
    def __init__(self, config):
        self.config = config
        self.dna_dict = {}
        self.dna = []
        self.simples = []  # [3, -1, 1701]
        self.dic = {}
        self.read_file(config.iran_2009_2010_path)
        # self.read_txt(config.guam_2009_txt_path)
        print("simple count : {count}".format(count=len(self.simples)))
        # self.pos = np.random.randint(0, self.num_examples)
        self.pos = 0

    def read_file(self, path):
        with open(path) as file:
            lines = file.readlines()
        new_line = []
        data = []
        key = ""
        for line in lines:
            if line.startswith(">gb"):
                key = line[4:12]
                continue
            if len(line) == 1:
                line = "".join(new_line)
                if len(line) == self.config.gene_length:
                    self.read_gene(line)
                    data.append(self.to_id(line))
                    self.dic[key] = self.to_id(line)
                    print("gene len : {len}".format(len=len(line)))
                new_line = []
            else:
                new_line.append(line.strip())
        print("simple len : {len}".format(len=len(data)))
        self.simples.append(data)

    def read_txt(self, path):
        with open(path) as file:
            lines = file.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if len(line) == 1734:
                line = line[13: 1714]
            if len(line) == self.config.gene_length:
                self.read_gene(line)
                data.append(self.to_id(line))
                print("gene len : {len}".format(len=len(line)))
        print("simple len : {len}".format(len=len(data)))
        self.simples.append(data)

    @property
    def num_examples(self):
        return len(self.simples)

    def next_batch(self, batch_size):
        next_pos = self.pos + 1
        if next_pos < self.num_examples:
            datas = self.simples[self.pos: next_pos]
        else:
            datas = self.simples[self.pos:]
            next_pos -= self.num_examples
            datas.extend(self.simples[: next_pos])
        self.pos = next_pos
        return datas

    def read_gene(self, gene):
        for dna in gene:
            if dna not in self.dna_dict:
                self.dna_dict[dna] = len(self.dna)
                self.dna.append(dna)

    def to_id(self, gene):
        """
        AGTC -> 1234
        """
        return [self.dna_dict[dna] for dna in gene]

    def to_gene(self, *ids):
        """
        1234 -> AGTC
        """
        result = [self.dna[i] for i in ids]
        return "".join(result)
