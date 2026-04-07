import os
import numpy as np
import paddle
from typing import List, Dict, Any, Tuple, Optional

class MACEDataset(paddle.io.Dataset):
    ELEMENT_TO_Z = {
        'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
        'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
        'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
        'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
        'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,
        'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
        'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,
        'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,
        'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,
        'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,
        'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,
        'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,
        'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,'Rf':104,'Db':105,
        'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,
        'Nh':113,'Fl':114,'Mc':115,'Lv':116,'Ts':117,'Og':118
    }

    def __init__(self, data_path: str, config: Any,
                 atomic_numbers: Optional[List[int]] = None,
                 processed_path: Optional[str] = None):
        super().__init__()
        self.data_path = data_path
        self.config = config
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        print(f"Loading raw data from {data_path}...")
        self.raw_data = self._load_raw_data()
        print(f"Loaded {len(self.raw_data)} structures.")
        if len(self.raw_data) == 0:
            raise ValueError(f"No data loaded from {data_path}")
        if atomic_numbers is None:
            atomic_numbers = self._get_atomic_numbers()
        self.atomic_numbers = atomic_numbers
        self.n_atom_types = len(atomic_numbers)
        self.atom_to_idx = {z: i for i, z in enumerate(atomic_numbers)}
        if processed_path and os.path.exists(processed_path):
            print(f"Loading preprocessed data from {processed_path}...")
            self.processed_data = paddle.load(processed_path)
            print("Done.")
        else:
            print("Preprocessing data (this may take a while)...")
            self.processed_data = self._preprocess()
            if processed_path:
                os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                print(f"Saving preprocessed data to {processed_path}...")
                paddle.save(self.processed_data, processed_path)
                print("Done.")
        print("Dataset ready.")

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        data = []
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            n_atoms = int(lines[i].strip())
            i += 1
            comment = lines[i].strip() if i < len(lines) else ""
            i += 1
            energy = None
            if 'energy=' in comment:
                try:
                    energy_part = comment.split('energy=')[1].split()[0]
                    energy = float(energy_part)
                except:
                    pass
            atomic_numbers = []
            positions = []
            forces = []
            for _ in range(n_atoms):
                if i >= len(lines):
                    break
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                parts = line.split()
                if len(parts) < 4:
                    i += 1
                    continue
                atom_type = parts[0]
                if atom_type.isalpha():
                    atomic_num = self.ELEMENT_TO_Z.get(atom_type, 0)
                else:
                    atomic_num = int(atom_type)
                atomic_numbers.append(atomic_num)
                try:
                    pos = [float(p) for p in parts[1:4]]
                    positions.append(pos)
                except:
                    i += 1
                    continue
                if len(parts) > 4:
                    try:
                        f = [float(p) for p in parts[4:7]]
                        forces.append(f)
                    except:
                        forces.append([0.,0.,0.])
                else:
                    forces.append([0.,0.,0.])
                i += 1
            if len(atomic_numbers) != n_atoms:
                continue
            data.append({
                'atomic_numbers': np.array(atomic_numbers, dtype=np.int64),
                'positions': np.array(positions, dtype=np.float32),
                'forces': np.array(forces, dtype=np.float32),
                'energy': energy
            })
        return data

    def _get_atomic_numbers(self) -> List[int]:
        atomic_nums = set()
        for d in self.raw_data:
            atomic_nums.update(d['atomic_numbers'])
        return sorted(list(atomic_nums))

    def _compute_neighbors(self, positions: np.ndarray, r_max: float
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_atoms = len(positions)
        edges = []
        dists = []
        vecs = []
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                delta = positions[j] - positions[i]
                d = np.linalg.norm(delta)
                if d < r_max and d > 1e-5:
                    edges.append([i, j])
                    edges.append([j, i])
                    dists.append(d)
                    dists.append(d)
                    vecs.append(delta / d)
                    vecs.append(-delta / d)
        if len(edges) == 0:
            return np.zeros((2,0), dtype=np.int64), np.zeros(0, dtype=np.float32), np.zeros((0,3), dtype=np.float32)
        edge_index = np.array(edges, dtype=np.int64).T
        edge_dist = np.array(dists, dtype=np.float32)
        edge_vec = np.array(vecs, dtype=np.float32)
        return edge_index, edge_dist, edge_vec

    def _preprocess(self) -> Dict[str, Any]:
        processed = []
        total = len(self.raw_data)
        for idx, data in enumerate(self.raw_data):
            atomic_numbers = data['atomic_numbers']
            positions = data['positions']
            edge_index, edge_dist, edge_vec = self._compute_neighbors(positions, self.config.r_max)
            atom_type_idx = np.array([self.atom_to_idx.get(z,0) for z in atomic_numbers], dtype=np.int64)
            processed.append({
                'atomic_numbers': atomic_numbers.astype(np.int64),
                'positions': positions.astype(np.float32),
                'atom_type_idx': atom_type_idx,
                'edge_index': edge_index,
                'edge_dist': edge_dist,
                'edge_vec': edge_vec,
                'energy': data['energy'],
                'forces': data['forces']
            })
            if (idx + 1) % 10000 == 0 or idx + 1 == total:
                print(f"  Preprocessed {idx+1}/{total} structures ({100*(idx+1)/total:.1f}%)")
        return {'data': processed}

    def __len__(self) -> int:
        return len(self.processed_data['data'])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.processed_data['data'][idx].copy()
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                if k in ['atomic_numbers', 'atom_type_idx', 'edge_index']:
                    item[k] = paddle.to_tensor(v, dtype='int64')
                else:
                    item[k] = paddle.to_tensor(v, dtype='float32')
            elif isinstance(v, (int,float)):
                item[k] = paddle.to_tensor([v], dtype='float32')
        return item
