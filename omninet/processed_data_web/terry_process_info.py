# this is terry's code for multi process generalization monitoring system

from copy import deepcopy
import re
import awkward as ak

PDG_Dict = {
  "Z": [23],
  "W": [24],
  "V": [23, 24],
  "l": [11, 13],
  "t": [6],
  "v": [12, 14, 16],
  "q": [1,2,3,4],
  "b": [5],
  "gamma": [22]
}

Feynman_diagram = {
  "TTHadronics": {"diagram": {"t1": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "t2": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "SYMMETRY": ["t1", "t2"]}},
  "TTHadronics_mlm": {"diagram": {"t1": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "t2": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "SYMMETRY": ["t1", "t2"]}},
  "GJets": {"diagram": {"gamma1": None}},
  "TT1L": {"diagram": {"t1": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "t2": {"b": None, "W": {"l": None, "v": None}}}},
  "TT1L_mlm": {"diagram": {"t1": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "t2": {"b": None, "W": {"l": None, "v": None}}}},
  "TT2L": {"diagram": {"t1": {"b": None, "W": {"l": None, "v": None}}, "t2": {"b": None, "W": {"l": None, "v": None}}, "SYMMETRY": ["t1", "t2"]}},
  "TT2L_mlm": {"diagram": {"t1": {"b": None, "W": {"l": None, "v": None}}, "t2": {"b": None, "W": {"l": None, "v": None}}, "SYMMETRY": ["t1", "t2"]}},
  "WJetsToLNu": {"diagram": {"W": {"l":None, "v": None}}},
  "WJetsToQQ": {"diagram": {"W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}},
  "ZJetsToQQ": {"diagram": {"Z": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}},
  "ZJetsToLL": {"diagram": {"Z": {"l1": None, "l2": None, "SYMMETRY": ["l1", "l2"]}}},
}


def select_by_pdgId(candidate, candidate_name):
    candidate_type = re.sub(r'\d+', '', candidate_name)
    candidate_pdgId = PDG_Dict[candidate_type]
    select = ak.values_astype(ak.zeros_like(candidate.pdgId), bool)
    for pdgId in candidate_pdgId:
        select = select | (abs(candidate.pdgId) == pdgId)
    return candidate[select]


def select_by_rank(Daughter, Mother, rank=0):
    order = ak.argsort(Mother.index)
    Daughter = Daughter[order]
    Mother = Mother[order]
    non_empty_mask = ak.num(Mother) > 0  # Check which sublists are non-empty
    run_length = ak.flatten((ak.run_lengths(Mother[non_empty_mask].index)))  # Filter out empty sublists

    Daughter = ak.unflatten(Daughter, counts = run_length, axis = 1)
    Mother  = ak.unflatten(Mother, counts = run_length, axis = 1)
    #Daughter = ak.pad_none(Daughter, rank + 1, axis = -1)
    #Mother = ak.pad_none(Mother, rank+1, axis = -1)
    Daughter = ak.flatten(Daughter[..., [rank]], axis = -1)
    Mother = ak.flatten(Mother[..., [rank]], axis = -1)
    return Daughter, Mother

def select_by_products(parton, candidate_array, products, candidate_name, process_summary = dict()):
    if "SYMMETRY" in products:
        symmetry = products["SYMMETRY"] 
        symmetry_map = {sym_: idx for idx, sym_ in enumerate(symmetry)}
    else:
        symmetry_map = None
    for product in products:
        if(product == "SYMMETRY"): continue
        product_name = '{}/{}'.format(candidate_name, product)
        product_array = select_by_pdgId(parton, product)
        cartesian = ak.argcartesian([product_array.M1, candidate_array.index], axis=1)
        matches = cartesian[(product_array.M1[cartesian["0"]] == candidate_array.index[cartesian["1"]])]

        product_from_mother = product_array[matches["0"]]
        candidate_array = candidate_array[matches["1"]]

        if symmetry_map is not None:
            #product_from_mother = ak.pad_none(product_from_mother, len(symmetry), axis=1)
            #candidate_array = ak.pad_none(candidate_array, len(symmetry), axis=1)
            product_from_mother, candidate_array = select_by_rank(product_from_mother, candidate_array, symmetry_map[product])
        candidate_array[product_name] = product_from_mother.index
        for i in product_from_mother[0]:
            print(i)
        print("----------")
        if products[product] is not None:
            product_decay_array, process_summary = select_by_products(parton, product_from_mother, products[product], product_name)
            print('---product decay---')
            for i in product_decay_array[1]:
                print(i)
            cartesian_selection = ak.argcartesian([product_decay_array.M1, candidate_array.index], axis=1)
            matches_selection = cartesian_selection[(product_decay_array.M1[cartesian_selection["0"]]==candidate_array.index[cartesian_selection["1"]])]
            product_selected = product_decay_array[matches_selection["0"]]
            candidate_array = candidate_array[matches_selection["1"]]
            for sub_product in product_selected.fields:
                if sub_product in candidate_array.fields: continue
                candidate_array[sub_product] = product_selected[sub_product]
        process_summary[product_name] = candidate_array[product_name]
        

    return candidate_array, process_summary

def assignment(candidate_array, products, candidate_name, process_summary = dict()):
    for product in products:
        if (product == "SYMMETRY"): continue
        product_name = '{}/{}'.format(candidate_name, product)
        process_summary[product_name] = candidate_array[product_name]
        if products[product] is not None:
            process_summary = assignment(candidate_array, products[product], product_name, process_summary)
    return process_summary

def assign_Reco_LorentzVector(Event_dict, products, parton, candidate_name, Reconstructed_momentum_dict = dict()):

  for product in products:
    if (product == 'SYMMETRY'): continue
    product_name = '{}/{}'.format(candidate_name, product)
    if (products[product] is not None):
      Reconstructed_momentum_dict = assign_Reco_LorentzVector(Event_dict, products[product], parton, product_name, Reconstructed_momentum_dict)
    else:
      Reconstructed_momentum_dict[product_name] = parton[parton.index == ak.unflatten(Event_dict[product_name], counts = 1, axis = -1)].reco_v4

    if candidate_name not in Reconstructed_momentum_dict:
      Reconstructed_momentum_dict[candidate_name] = Reconstructed_momentum_dict[product_name]
    else:
      Reconstructed_momentum_dict[candidate_name] = ak.where((Reconstructed_momentum_dict[product_name].pt > 0) & (Reconstructed_momentum_dict[candidate_name].pt > 0),
                                                             (Reconstructed_momentum_dict[candidate_name].add(Reconstructed_momentum_dict[product_name])),
                                                             (ak.ones_like(Reconstructed_momentum_dict[product_name])*-1))
 
  return Reconstructed_momentum_dict


