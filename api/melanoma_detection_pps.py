import pickle
import numpy as np

TUMOR_CODES = {'primary': 0,
               'NS': 1,
               'metastasis': 2,
               'secondary': 3,
               'recurrent': 4,
               'hyperplasia adjacent to primary tumour': 5}

IS_MELANOMA_CODES = {0: 'No', 1: 'Yes'}

GENE_CODES = {'BRAF': 0, 'LRP1B': 1, 'MUC16': 2, 'NRAS': 2}

BRAF_dist = {'M': 18,
             'A': 52,
             'L': 68,
             'S': 78,
             'G': 56,
             'E': 42,
             'P': 51,
             'Q': 41,
             'F': 33,
             'N': 27,
             'D': 39,
             'I': 43,
             'V': 40,
             'W': 8,
             'K': 41,
             'T': 39,
             'H': 20,
             'Y': 17,
             'R': 40,
             'C': 13}

MUC16_dist = {'M': 358,
              'L': 1100,
              'K': 359,
              'P': 1250,
              'S': 2647,
              'G': 752,
              'T': 2571,
              'R': 459,
              'A': 789,
              'E': 807,
              'D': 456,
              'I': 571,
              'V': 783,
              'H': 310,
              'F': 316,
              'Q': 320,
              'N': 354,
              'Y': 171,
              'W': 98,
              'C': 36}

LRP1B_dist = {'M': 68,
              'S': 318,
              'E': 250,
              'F': 139,
              'L': 314,
              'A': 200,
              'T': 249,
              'G': 356,
              'P': 185,
              'I': 277,
              'R': 236,
              'V': 216,
              'D': 405,
              'Q': 152,
              'C': 345,
              'H': 131,
              'W': 88,
              'K': 223,
              'N': 290,
              'Y': 157}

aminos = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

model = pickle.load(open('model/pps_gbc.pkl', 'rb'))


def amino_acid_distribution(seq):
    """
    Convert PPS sequence into amino acid count distribution
    :param seq: PPS as a sequence of amino acids with string format
    :return: PPS amino acid count distribution as a dict
    """
    dic = dict()
    for ch in seq:
        if ch not in dic.keys():
            dic[ch] = 0
        dic[ch] += 1
    return dic


def get_pps_changes(gene_name, row):
    """
    Derive changes in PPS using original and mutated PPS
    :param gene_name: Gene name as instance of string
    :param row: mutated PPS CD
    :return: PPS changes as dict
    """
    if gene_name == 'BRAF':
        dist = BRAF_dist.copy()
    elif gene_name == 'MUC16':
        dist = MUC16_dist.copy()
    elif gene_name == 'LRP1B':
        dist = LRP1B_dist.copy()
    else:
        raise Exception(f'Unknown gene name : %s', gene_name)

    keys = row.keys()
    for amino in aminos:
        if amino not in keys:
            row[amino] = 0
        row[amino] = dist[amino] - row[amino]

    return row


class Data:
    def __init__(self, age, gene_name, tumour_origin, tier, mutated_dna_seq):
        """
        :param age:
        :param gene_name:
        :param tumour_origin:
        :param tier:
        :param mutated_dna_seq:
        """
        self.gene_name = gene_name
        self.gene_code = GENE_CODES[gene_name]
        self.age = age
        self.tumour_origin = tumour_origin
        self.tumour_origin_code = TUMOR_CODES[tumour_origin]
        self.tier = tier
        self.pps = list(get_pps_changes(gene_name, amino_acid_distribution(dna_to_protein(mutated_dna_seq))).values())

    def list(self):
        """
        Format to list that requires for input to the model
        :return: returns data as list
        """
        lst = [self.gene_code, self.age, self.tumour_origin_code, self.tier]
        lst.extend(self.pps)
        return lst

    def log(self):
        """
        Log details
        :return: None
        """
        print("---> Patient Details")
        print(f"\tGene Code : {self.gene_code}")
        print(f"\tAge : {self.age}")
        print(f"\tTumor Origin Code : {self.tumour_origin_code}")
        print(f"\tTier : {self.tier}")
        print(f"\tPPS : {self.pps}")
        lst = [self.gene_code, self.age, self.tumour_origin_code, self.tier]
        lst.extend(self.pps)
        print(f"Dataframe : {lst}")


def detect_melanoma_by_pps(pps_data):
    """
    Detect melanoma
    :param pps_data: instance of Data
    :return: returns Yes/No
    """
    pps_data.log()
    _input = [np.array(pps_data.list())]
    probability = model.predict_proba(_input)
    res = {"gene": pps_data.gene_name, "tumor": pps_data.tumour_origin, "tier": pps_data.tier,
           "age": pps_data.age, "pps": pps_data.pps, "probability": probability[0][1]}
    return res


# define codon table
protein = {"TTT": "F", "CTT": "L", "ATT": "I", "GTT": "V",
           "TTC": "F", "CTC": "L", "ATC": "I", "GTC": "V",
           "TTA": "L", "CTA": "L", "ATA": "I", "GTA": "V",
           "TTG": "L", "CTG": "L", "ATG": "M", "GTG": "V",
           "TCT": "S", "CCT": "P", "ACT": "T", "GCT": "A",
           "TCC": "S", "CCC": "P", "ACC": "T", "GCC": "A",
           "TCA": "S", "CCA": "P", "ACA": "T", "GCA": "A",
           "TCG": "S", "CCG": "P", "ACG": "T", "GCG": "A",
           "TAT": "Y", "CAT": "H", "AAT": "N", "GAT": "D",
           "TAC": "Y", "CAC": "H", "AAC": "N", "GAC": "D",
           "TAA": "STOP", "CAA": "Q", "AAA": "K", "GAA": "E",
           "TAG": "STOP", "CAG": "Q", "AAG": "K", "GAG": "E",
           "TGT": "C", "CGT": "R", "AGT": "S", "GGT": "G",
           "TGC": "C", "CGC": "R", "AGC": "S", "GGC": "G",
           "TGA": "STOP", "CGA": "R", "AGA": "R", "GGA": "G",
           "TGG": "W", "CGG": "R", "AGG": "R", "GGG": "G"
           }


def dna_to_protein(dna):
    """
        This function takes dna sequence as string
        and convert into amino acid sequence
    """

    protein_sequence = ""

    # Generate protein sequence
    for i in range(0, len(dna) - (3 + len(dna) % 3), 3):
        if protein[dna[i:i + 3]] == "STOP":
            break
        protein_sequence += protein[dna[i:i + 3]]

    return protein_sequence

# mutatedDNAEx1 = "ATGGCGGCGCTGAGCGGTGGCGGTGGTGGCGGCGCGGAGCCGGGCCAGGCTCTGTTCAACGGGGACATGGAGCCCGAGGCCGGCGCCGGCGCCGGCGCCGCGGCCTCTTCGGCTGCGGACCCTGCCATTCCGGAGGAGGTGTGGAATATCAAACAAATGATTAAGTTGACACAGGAACATATAGAGGCCCTATTGGACAAATTTGGTGGGGAGCATAATCCACCATCAATATATCTGGAGGCCTATGAAGAATACACCAGCAAGCTAGATGCACTCCAACAAAGAGAACAACAGTTATTGGAATCTCTGGGGAACGGAACTGATTTTTCTGTTTCTAGCTCTGCATCAATGGATACCGTTACATCTTCTTCCTCTTCTAGCCTTTCAGTGCTACCTTCATCTCTTTCAGTTTTTCAAAATCCCACAGATGTGGCACGGAGCAACCCCAAGTCACCACAAAAACCTATCGTTAGAGTCTTCCTGCCCAACAAACAGAGGACAGTGGTACCTGCAAGGTGTGGAGTTACAGTCCAAGACAGTCTAAAGAAAGCACTGATGATGAGAGGTCTAATCCCAGAGTGCTGTGCTGTTTACAGAATTCAGGATGGAGAGAAGAAACCAATTGGTTGGGACACTGATATTTCCTGGCTTACTGGAGAAGAATTGCATGTGGAAGTGTTGGAGAATGTTCCACTTACAACACACAACTTTGTACGAAAAACGTTTTTCACCTTAGCATTTTGTGACTTTTGTCGAAAGCTGCTTTTCCAGGGTTTCCGCTGTCAAACATGTGGTTATAAATTTCACCAGCGTTGTAGTACAGAAGTTCCACTGATGTGTGTTAATTATGACCAACTTGATTTGCTGTTTGTCTCCAAGTTCTTTGAACACCACCCAATACCACAGGAAGAGGCGTCCTTAGCAGAGACTGCCCTAACATCTGGATCATCCCCTTCCGCACCCGCCTCGGACTCTATTGGGCCCCAAATTCTCACCAGTCCGTCTCCTTCAAAATCCATTCCAATTCCACAGCCCTTCCGACCAGCAGATGAAGATCATCGAAATCAATTTGGGCAACGAGACCGATCCTCATCAGCTCCCAATGTGCATATAAACACAATAGAACCTGTCAATATTGATGACTTGATTAGAGACCAAGGATTTCGTGGTGATGGAGGATCAACCACAGGTTTGTCTGCTACCCCCCCTGCCTCATTACCTGGCTCACTAACTAACGTGAAAGCCTTACAGAAATCTCCAGGACCTCAGCGAGAAAGGAAGTCATCTTCATCCTCAGAAGACAGGAATCGAATGAAAACACTTGGTAGACGGGACTCGAGTGATGATTGGGAGATTCCTGATGGGCAGATTACAGTGGGACAAAGAATTGGATCTGGATCATTTGGAACAGTCTACAAGGGAAAGTGGCATGGTGATGTGGCAGTGAAAATGTTGAATGTGACAGCACCTACACCTCAGCAGTTACAAGCCTTCAAAAATGAAGTAGGAGTACTCAGGAAAACACGACATGTGAATATCCTACTCTTCATGGGCTATTCCACAAAGCCACAACTGGCTATTGTTACCCAGTGGTGTGAGGGCTCCAGCTTGTATCACCATCTCCATATCATTGAGACCAAATTTGAGATGATCAAACTTATAGATATTGCACGACAGACTGCACAGGGCATGGATTACTTACACGCCAAGTCAATCATCCACAGAGACCTCAAGAGTAATAATATATTTCTTCATGAAGACCTCACAGTAAAAATAGGTGATTTTGGTCTAGCTACAGTGAAATCTCGATGGAGTGGGTCCCATCAGTTTGAACAGTTGTCTGGATCCATTTTGTGGATGGCACCAGAAGTCATCAGAATGCAAGATAAAAATCCATACAGCTTTCAGTCAGATGTATATGCATTTGGAATTGTTCTGTATGAATTGATGACTGGACAGTTACCTTATTCAAACATCAACAACAGGGACCAGATAATTTTTATGGTGGGACGAGGATACCTGTCTCCAGATCTCAGTAAGGTACGGAGTAACTGTCCAAAAGCCATGAAGAGATTAATGGCAGAGTGCCTCAAAAAGAAAAGAGATGAGAGACCACTCTTTCCCCAAATTCTCGCCTCTATTGAGCTGCTGGCCCGCTCATTGCCAAAAATTCACCGCAGTGCATCAGAACCCTCCTTGAATCGGGCTGGTTTCCAAACAGAGGATTTTAGTCTATATGCTTGTGCTTCTCCAAAAACACCCATCCAGGCAGGGGGATATGGTGCGTTTCCTGTCCACTGA "
# mutatedDNAEx2 = "ATGGCGGGCTGAGCGGTGGCGGTGGTGGCGGCGCGGAGCCGGGCCAGGCTCTGTTCAACGGGGACATGGAGCCCGAGGCCGGCGCCGGCGCCGGCGCCGCGGCCTCTTCGGCTGCGGACCCTGCCATTCCGGAGGAGGTGTGGAATATCAAACAAATGATTAAGTTGACACAGGAACATATAGAGGCCCTATTGGACAAATTTGGTGGGGAGCATAATCCACCATCAATATATCTGGAGGCCTATGAAGAATACACCAGCAAGCTAGATGCACTCCAACAAAGAGAACAACAGTTATTGGAATCTCTGGGGAACGGAACTGATTTTTCTGTTTCTAGCTCTGCATCAATGGATACCGTTACATCTTCTTCCTCTTCTAGCCTTTCAGTGCTACCTTCATCTCTTTCAGTTTTTCAAAATCCCACAGATGTGGCACGGAGCAACCCCAAGTCACCACAAAAACCTATCGTTAGAGTCTTCCTGCCCAACAAACAGAGGACAGTGGTACCTGCAAGGTGTGGAGTTACAGTCCAAGACAGTCTAAAGAAAGCACTGATGATGAGAGGTCTAATCCCAGAGTGCTGTGCTGTTTACAGAATTCAGGATGGAGAGAAGAAACCAATTGGTTGGGACACTGATATTTCCTGGCTTACTGGAGAAGAATTGCATGTGGAAGTGTTGGAGAATGTTCCACTTACAACACACAACTTTGTACGAAAAACGTTTTTCACCTTAGCATTTTGTGACTTTTGTCGAAAGCTGCTTTTCCAGGGTTTCCGCTGTCAAACATGTGGTTATAAATTTCACCAGCGTTGTAGTACAGAAGTTCCACTGATGTGTGTTAATTATGACCAACTTGATTTGCTGTTTGTCTCCAAGTTCTTTGAACACCACCCAATACCACAGGAAGAGGCGTCCTTAGCAGAGACTGCCCTAACATCTGGATCATCCCCTTCCGCACCCGCCTCGGACTCTATTGGGCCCCAAATTCTCACCAGTCCGTCTCCTTCAAAATCCATTCCAATTCCACAGCCCTTCCGACCAGCAGATGAAGATCATCGAAATCAATTTGGGCAACGAGACCGATCCTCATCAGCTCCCAATGTGCATATAAACACAATAGAACCTGTCAATATTGATGACTTGATTAGAGACCAAGGATTTCGTGGTGATGGAGGATCAACCACAGGTTTGTCTGCTACCCCCCCTGCCTCATTACCTGGCTCACTAACTAACGTGAAAGCCTTACAGAAATCTCCAGGACCTCAGCGAGAAAGGAAGTCATCTTCATCCTCAGAAGACAGGAATCGAATGAAAACACTTGGTAGACGGGACTCGAGTGATGATTGGGAGATTCCTGATGGGCAGATTACAGTGGGACAAAGAATTGGATCTGGATCATTTGGAACAGTCTACAAGGGAAAGTGGCATGGTGATGTGGCAGTGAAAATGTTGAATGTGACAGCACCTACACCTCAGCAGTTACAAGCCTTCAAAAATGAAGTAGGAGTACTCAGGAAAACACGACATGTGAATATCCTACTCTTCATGGGCTATTCCACAAAGCCACAACTGGCTATTGTTACCCAGTGGTGTGAGGGCTCCAGCTTGTATCACCATCTCCATATCATTGAGACCAAATTTGAGATGATCAAACTTATAGATATTGCACGACAGACTGCACAGGGCATGGATTACTTACACGCCAAGTCAATCATCCACAGAGACCTCAAGAGTAATAATATATTTCTTCATGAAGACCTCACAGTAAAAATAGGTGATTTTGGTCTAGCTACAGTGAAATCTCGATGGAGTGGGTCCCATCAGTTTGAACAGTTGTCTGGATCCATTTTGTGGATGGCACCAGAAGTCATCAGAATGCAAGATAAAAATCCATACAGCTTTCAGTCAGATGTATATGCATTTGGAATTGTTCTGTATGAATTGATGACTGGACAGTTACCTTATTCAAACATCAACAACAGGGACCAGATAATTTTTATGGTGGGACGAGGATACCTGTCTCCAGATCTCAGTAAGGTACGGAGTAACTGTCCAAAAGCCATGAAGAGATTAATGGCAGAGTGCCTCAAAAAGAAAAGAGATGAGAGACCACTCTTTCCCCAAATTCTCGCCTCTATTGAGCTGCTGGCCCGCTCATTGCCAAAAATTCACCGCAGTGCATCAGAACCCTCCTTGAATCGGGCTGGTTTCCAAACAGAGGATTTTAGTCTATATGCTTGTGCTTCTCCAAAAACACCCATCCAGGCAGGGGGATATGGTGCGTTTCCTGTCCACTGA "


# if __name__ == '__main__':
#     data = Data(20, 'BRAF', 'metastasis', 2, mutatedDNAEx1)
#     data.log()
#     print(detect_melanoma_by_pps(data))
