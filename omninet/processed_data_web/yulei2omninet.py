import h5py
import numpy as np


# read yulei data
yulei_file_path = '/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/run_yulei_2/TTHadronics_367772000.h5'
out_file_path = '/pscratch/sd/w/weipow/OmniNet_Data/TTHadronics_367772000_omninet_10jets.h5'

# Open yulei generated file and my output file
with h5py.File(yulei_file_path, 'r') as infile, h5py.File(out_file_path, 'w') as outfile:
    
    # Create new groups for INPUTS and TARGETS
    inputs_group = outfile.create_group("INPUTS")
    source_group = inputs_group.create_group("Source")
    
    # Create new groups for TARGETS
    targets_group = outfile.create_group("TARGETS")
    t1_group = targets_group.create_group("t1")
    t2_group = targets_group.create_group("t2")
    
    # # Add datasets for INPUTS/Source 
    # # # Get data from infile[jets]: ["jet_pt","jet_eta","jet_phi","jet_m","jet_btag","jet_npart","jet_flavor"]
    jets_data = infile['jets'][:]
    print("jets dataset:", jets_data.shape) # (10000, 4, 7)

    # jets_pt = jets_data[:,:,0]
    # jets_eta = jets_data[:,:,1]
    # jets_phi = jets_data[:,:,2]
    # jets_mass = jets_data[:,:,3]
    # jets_btag = jets_data[:,:,4]

    # # # Write the data into out file
    # source_group.create_dataset("pt", data=jets_pt.astype('<f4'))
    # source_group.create_dataset("eta", data=jets_eta.astype('<f4'))
    # source_group.create_dataset("phi", data=jets_phi.astype('<f4'))
    # source_group.create_dataset("mass", data=jets_mass.astype('<f4'))
    # source_group.create_dataset("btag", data=jets_btag.astype('<f4'))

    # # MASK: if the jets is padded or not. (You may check, but I remember True means this jet is physical and False means this jet is padded one)
    # mask_data = np.full((10000, 4), True, dtype='|b1')
    # source_group.create_dataset("MASK", data=mask_data, dtype='|b1')


    
    # Add datasets for TARGETS/t1, t2
    # # Get data from infile["genpart"]: ["genpart_pt", "genpart_eta", "genpart_phi", "genpart_m", "genpart_index", "genpart_M1", "genpart_M2", "genpart_PID", "genpart_Status", "genmatched_index"]
    genpart_data = infile['genpart'][:]
    print("genpart_data dataset:", genpart_data.shape) # (10000, 12, 10)

    genpart_index = genpart_data[:,:,4]
    genpart_M1 = genpart_data[:,:,5]
    genpart_PID = genpart_data[:,:,7]
    genmatched_index = genpart_data[:,:,-1]

    # # temporary: for matched > 4, make it to -1
    # genmatched_index[genmatched_index>4] = -1

    # loop over all events
    n_evt = genpart_data.shape[0]

    # # INPUT
    jets_pt_data = []
    jets_eta_data = []
    jets_phi_data = []
    jets_mass_data = []
    jets_btag_data = []
    mask_data = []

    # # TARGET
    t1_b_data = []
    t1_q1_data = []
    t1_q2_data = []
    t2_b_data = []
    t2_q1_data = []
    t2_q2_data = []
    count = 0


    for evt in range(n_evt):
        # check if the b is from top, if not reject event
        # get b mother index 
        b_genpart_M1 = genpart_M1[evt][(abs(genpart_PID[evt])==5)]
        mother_mask = (genpart_index[evt]==b_genpart_M1[0]) | (genpart_index[evt]==b_genpart_M1[1])
        mother_pid = genpart_PID[evt][mother_mask]

        # pid of top is 6
        if (abs(mother_pid[0]) == 6) and (abs(mother_pid[1]) == 6):

            # genmatched_index for b
            b_genmatched_index = genmatched_index[evt][(abs(genpart_PID[evt])==5)]
            # gen_matched_index for ucds
            q_genmatched_index = genmatched_index[evt][(abs(genpart_PID[evt])==1) | (abs(genpart_PID[evt])==2) | (abs(genpart_PID[evt])==3) | (abs(genpart_PID[evt])==4)]

            # check double assignment
            if b_genmatched_index[0] == b_genmatched_index[1]:
                print("double assignment for b!")
                continue

            t1_b_data.append(b_genmatched_index[0])
            t2_b_data.append(b_genmatched_index[1])

            t1_q1_data.append(q_genmatched_index[0])
            t1_q2_data.append(q_genmatched_index[1])
            t2_q1_data.append(q_genmatched_index[2])
            t2_q2_data.append(q_genmatched_index[3])

            jets_pt_data.append(jets_data[evt,:,0])
            jets_eta_data.append(jets_data[evt,:,1])
            jets_phi_data.append(jets_data[evt,:,2])
            jets_mass_data.append(jets_data[evt,:,3])
            jets_btag_data.append(jets_data[evt,:,4])
            mask_data.append(np.full((10), True, dtype='|b1'))

            count+=1


    source_group.create_dataset("pt", data=np.array(jets_pt_data).astype('<f4'))
    source_group.create_dataset("eta", data=np.array(jets_eta_data).astype('<f4'))
    source_group.create_dataset("phi", data=np.array(jets_phi_data).astype('<f4'))
    source_group.create_dataset("mass", data=np.array(jets_mass_data).astype('<f4'))
    source_group.create_dataset("btag", data=np.array(jets_btag_data).astype('<f4'))
    source_group.create_dataset("MASK", data=np.array(mask_data), dtype='|b1')

    t1_group.create_dataset("b", data=np.array(t1_b_data).astype('<i8'))
    t1_group.create_dataset("q1", data=np.array(t1_q1_data).astype('<i8'))
    t1_group.create_dataset("q2", data=np.array(t1_q2_data).astype('<i8'))
    t2_group.create_dataset("b", data=np.array(t2_b_data).astype('<i8'))
    t2_group.create_dataset("q1", data=np.array(t2_q1_data).astype('<i8'))
    t2_group.create_dataset("q2", data=np.array(t2_q2_data).astype('<i8'))

    print(count)

# output yulei_omninet.h5