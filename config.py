CHAR_LIST = [" ", "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Ti",
             "V", "Cr", "Mn", "Fe", "Ni", "Cu", "Zn", "Ge", "As", "Se", "Br", "Sr", "Zr", "Mo", "Pd", "Yb", "Ag", "Cd",
             "Sb", "I", "Ba", "Nd", "Gd", "Dy", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
             "n", "c", "o", "s", "se",
             "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "(", ")", "[", "]",
             "-", "=", "#", "/", "\\", "+", "@", "."]
CHAR_LEN = len(CHAR_LIST)
TOX21_TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
               'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
MAX_SEQ_LEN = 200
CUDA_VISIBLE_DEVICES = 0

BATCH = 128
ADDENNUM = 300

CYCLES = 5

LATENT_DIM = 300
LR = 1e-3
MOMENTUM = 0.9
WDECAY = 1e-5
MILESTONES = [160, 240]
EPOCHL = 120
MARGIN = 1.0  # xi
WEIGHT = 1.0  # lambda
BETA = 1
