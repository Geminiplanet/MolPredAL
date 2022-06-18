TOX21_CHAR_LIST = [" ", "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Ti",
             "V", "Cr", "Mn", "Fe", "Ni", "Cu", "Zn", "Ge", "As", "Se", "Br", "Sr", "Zr", "Mo", "Pd", "Yb", "Ag", "Cd",
             "Sb", "I", "Ba", "Nd", "Gd", "Dy", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
             "n", "c", "o", "s", "se",
             "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "(", ")", "[", "]",
             "-", "=", "#", "/", "\\", "+", "@", "."]
QM9_CHAR_LIST = [" ", "H", "C", "N", "O", "F",
                 "1", "2", "3", "4", "5",
                 "(", ")", "[", "]",
                 "-", "=", "#", ":", "/", "\\", "+"]
TOX21_TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
               'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
QM9_TASKS = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
MAX_TOX21_LEN = 200
MAX_QM9_LEN = 42

CUDA_VISIBLE_DEVICES = 0

BATCH = 100
ADDENNUM = 200
SUBSET = 5000

CYCLES = 10

LATENT_DIM = 300
LR = 5e-4
MOMENTUM = 0.9
WDECAY = 1e-5
MILESTONES = [160, 240]
EPOCHL = 120
MARGIN = 1.0  # xi
WEIGHT = 1.0  # lambda
BETA = 1
