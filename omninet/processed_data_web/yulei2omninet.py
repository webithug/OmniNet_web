import h5py
import numpy as np


def has_double_assignment(array):
    count_dict = {}
    
    for num in array:
        if num != -1:
            if num in count_dict:
                return True
            count_dict[num] = 1
            
    return False

# "TTHadronics": {"diagram": {"t1": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "t2": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "SYMMETRY": ["t1", "t2"]}},
def convert_TTHadronics(yulei_file_path, out_file_path):

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


        # loop over each event
        for evt in range(n_evt):

            # check double assignment
            if has_double_assignment(genmatched_index[evt]):
                continue

            # get b mother pid
            b_genpart_M1 = genpart_M1[evt][(abs(genpart_PID[evt])==5)] # get b mother index 
            b_mother_mask = (genpart_index[evt]==b_genpart_M1[0]) | (genpart_index[evt]==b_genpart_M1[1]) # find location of b_mother in the array
            b_mother_pid = genpart_PID[evt][b_mother_mask]
            # print(b_genpart_M1)

            
            # check if the b is from top, if not reject event. pid of top is 6
            if (abs(b_mother_pid[0]) == 6) and (abs(b_mother_pid[1]) == 6):

                # genpart_index for b
                b_genpart_indices = genpart_index[evt][(abs(genpart_PID[evt])==5)]
                t1_b_index = b_genpart_indices[0]
                t2_b_index = b_genpart_indices[1]


                t1_b_data.append(genmatched_index[evt][genpart_index[evt]==t1_b_index].item())
                t2_b_data.append(genmatched_index[evt][genpart_index[evt]==t2_b_index].item())


                # find t1_q1 and t1_q2, they should come from a W with W_M1==b_genpart_M1[0]
                t1_idx = b_genpart_M1[0]
                mom_is_t1_mask = genpart_M1[evt]==t1_idx
                # get index of t1_W (both W and b have same mom t1)
                if genpart_index[evt][mom_is_t1_mask][0]==t1_b_index:
                    t1_W_idx = genpart_index[evt][mom_is_t1_mask][1]
                else :
                    t1_W_idx = genpart_index[evt][mom_is_t1_mask][0]
                # mask for t1_q1 and t1_q2
                t1_W_qs_mask = (genpart_M1[evt]==t1_W_idx)

                t1_q1_data.append(genmatched_index[evt][t1_W_qs_mask][0].item())
                t1_q2_data.append(genmatched_index[evt][t1_W_qs_mask][1].item())


                # find t2_q1 and t2_q2, they should come from a W with W_M1==b_genpart_M1[1]
                t2_idx = b_genpart_M1[1]
                mom_is_t2_mask = genpart_M1[evt]==t2_idx
                # get index of t2_W
                if genpart_index[evt][mom_is_t2_mask][0]==t2_b_index:
                    t2_W_idx = genpart_index[evt][mom_is_t2_mask][1]
                else :
                    t2_W_idx = genpart_index[evt][mom_is_t2_mask][0]
                # mask for t1_q1 and t1_q2
                t2_W_qs_mask = (genpart_M1[evt]==t2_W_idx)

                t2_q1_data.append(genmatched_index[evt][t2_W_qs_mask][0].item())
                t2_q2_data.append(genmatched_index[evt][t2_W_qs_mask][1].item())


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

# "TT2L": {"diagram": {"t1": {"b": None, "W": {"l": None, "v": None}}, "t2": {"b": None, "W": {"l": None, "v": None}}, "SYMMETRY": ["t1", "t2"]}}
def convert_TT2L(yulei_file_path, out_file_path):
    # Open yulei generated file and my output file
    with h5py.File(yulei_file_path, 'r') as infile, h5py.File(out_file_path, 'w') as outfile:
        
        # Create new groups for INPUTS
        inputs_group = outfile.create_group("INPUTS")
        source_group = inputs_group.create_group("Source")
        
        # Create new groups for TARGETS
        targets_group = outfile.create_group("TARGETS")
        t1_group = targets_group.create_group("t1")
        t2_group = targets_group.create_group("t2")
        
        # # Add datasets for INPUTS/Source 
        # # # Get data from infile[jets]: ["jet_pt","jet_eta","jet_phi","jet_m","jet_btag","jet_npart","jet_flavor"]
        jets_data = infile['jets'][:]
        print("jets dataset:", jets_data.shape) # (10000, 10, 7)
        # # # Get data from infile[els]: ["el_pt","el_eta","el_phi","el_m","el_ch"]
        els_data = infile['els'][:]
        print("els dataset:", els_data.shape) # (10000, 4, 5)
        # # # Get data from infile[mus]: ["mu_pt","mu_eta","mu_phi","mu_m","mu_ch"]
        mus_data = infile['mus'][:]
        print("mus dataset:", mus_data.shape) # (10000, 4, 5)

        # Add datasets for TARGETS/t1, t2
        # # Get data from infile["genpart"]: ["genpart_pt", "genpart_eta", "genpart_phi", "genpart_m", "genpart_index", "genpart_M1", "genpart_M2", "genpart_PID", "genpart_Status", "genmatched_index"]
        genpart_data = infile['genpart'][:]
        print("genpart_data dataset:", genpart_data.shape) # (10000, 12, 10)

        genpart_index = genpart_data[:,:,4]
        genpart_M1 = genpart_data[:,:,5]
        genpart_PID = genpart_data[:,:,7]
        genmatched_index = genpart_data[:,:,-1]




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
        t1_l_data = []
        t1_v_data = []
        t2_b_data = []
        t2_l_data = []
        t2_v_data = []
        count = 0


        # loop over each event
        for evt in range(n_evt):

            # check double assignment
            if has_double_assignment(genmatched_index[evt]):
                continue

            # get b mother pid
            b_genpart_M1 = genpart_M1[evt][(abs(genpart_PID[evt])==5)] # get b mother index 
            b_mother_mask = (genpart_index[evt]==b_genpart_M1[0]) | (genpart_index[evt]==b_genpart_M1[1]) # find location of b_mother in the array
            b_mother_pid = genpart_PID[evt][b_mother_mask]
            # print(b_genpart_M1)

            
            # check if the b is from top, if not reject event. (pid of top is 6)
            if (abs(b_mother_pid[0]) == 6) and (abs(b_mother_pid[1]) == 6):

                # genpart_index for b
                b_genpart_indices = genpart_index[evt][(abs(genpart_PID[evt])==5)]
                t1_b_index = b_genpart_indices[0]
                t2_b_index = b_genpart_indices[1]



                # find t1_l and t1_v, they should come from a W with W_M1==b_genpart_M1[0]
                t1_idx = b_genpart_M1[0]
                mom_is_t1_mask = genpart_M1[evt]==t1_idx
                # get index of t1_W (both W and b have same mom t1)
                if genpart_index[evt][mom_is_t1_mask][0]==t1_b_index:
                    t1_W_idx = genpart_index[evt][mom_is_t1_mask][1]
                else :
                    t1_W_idx = genpart_index[evt][mom_is_t1_mask][0]

                # mask to find parts whose mother is t1_W, this should give us t1_l and t1_v
                t1_W_lv_mask = (genpart_M1[evt]==t1_W_idx)



                # find t2_l and t2_v, they should come from a W with W_M1==b_genpart_M1[1]
                t2_idx = b_genpart_M1[1]
                mom_is_t2_mask = genpart_M1[evt]==t2_idx
                # get index of t2_W
                if genpart_index[evt][mom_is_t2_mask][0]==t2_b_index:
                    t2_W_idx = genpart_index[evt][mom_is_t2_mask][1]
                else :
                    t2_W_idx = genpart_index[evt][mom_is_t2_mask][0]

                # mask to find parts whose mother is t2_W, this should give us t2_l and t2_v
                t2_W_lv_mask = (genpart_M1[evt]==t2_W_idx)



                # start filling data
                # check if each evt has 2 parts whose mother is t1_W and 2 for t2_W, else reject evt
                if ( (len(genmatched_index[evt][t1_W_lv_mask]) == 2) and (len(genmatched_index[evt][t2_W_lv_mask]) == 2) ):

                    # fill the t1 data
                    t1_b_data.append(genmatched_index[evt][genpart_index[evt]==t1_b_index].item())
                    # distinguish t1_l and t1_v
                    if abs(genpart_PID[evt][t1_W_lv_mask][0])== 11: # [0] is electron, gen_match_index: 10~13
                        t1_l_data.append(genmatched_index[evt][t1_W_lv_mask][0].item()+10 if genmatched_index[evt][t1_W_lv_mask][0].item() >= 0 else genmatched_index[evt][t1_W_lv_mask][0].item())
                        t1_v_data.append(genmatched_index[evt][t1_W_lv_mask][1].item())
                    elif abs(genpart_PID[evt][t1_W_lv_mask][0])== 13: # [0] is muon, gen_match_index: 14~17
                        t1_l_data.append(genmatched_index[evt][t1_W_lv_mask][0].item()+14 if genmatched_index[evt][t1_W_lv_mask][0].item() >= 0 else genmatched_index[evt][t1_W_lv_mask][0].item())
                        t1_v_data.append(genmatched_index[evt][t1_W_lv_mask][1].item())
                    elif abs(genpart_PID[evt][t1_W_lv_mask][1])== 11: # [1] is electron, gen_match_index: 10~13
                        t1_l_data.append(genmatched_index[evt][t1_W_lv_mask][1].item()+10 if genmatched_index[evt][t1_W_lv_mask][0].item() >= 0 else genmatched_index[evt][t1_W_lv_mask][0].item())
                        t1_v_data.append(genmatched_index[evt][t1_W_lv_mask][0].item())
                    else: # [1] is muon, gen_match_index: 14~17
                        t1_l_data.append(genmatched_index[evt][t1_W_lv_mask][1].item()+14 if genmatched_index[evt][t1_W_lv_mask][0].item() >= 0 else genmatched_index[evt][t1_W_lv_mask][0].item())
                        t1_v_data.append(genmatched_index[evt][t1_W_lv_mask][0].item())

                    # fill in t2 data
                    t2_b_data.append(genmatched_index[evt][genpart_index[evt]==t2_b_index].item())
                    # distinguish t2_l and t2_v
                    if abs(genpart_PID[evt][t2_W_lv_mask][0])== 11:
                        t2_l_data.append(genmatched_index[evt][t2_W_lv_mask][0].item()+10 if genmatched_index[evt][t2_W_lv_mask][0].item() >= 0 else genmatched_index[evt][t2_W_lv_mask][0].item())
                        t2_v_data.append(genmatched_index[evt][t2_W_lv_mask][1].item())
                    elif abs(genpart_PID[evt][t2_W_lv_mask][0])== 13:
                        t2_l_data.append(genmatched_index[evt][t2_W_lv_mask][0].item()+14 if genmatched_index[evt][t2_W_lv_mask][0].item() >= 0 else genmatched_index[evt][t2_W_lv_mask][0].item())
                        t2_v_data.append(genmatched_index[evt][t2_W_lv_mask][1].item())
                    elif abs(genpart_PID[evt][t2_W_lv_mask][1])== 11: 
                        t2_l_data.append(genmatched_index[evt][t2_W_lv_mask][1].item()+10 if genmatched_index[evt][t2_W_lv_mask][0].item() >= 0 else genmatched_index[evt][t2_W_lv_mask][0].item())
                        t2_v_data.append(genmatched_index[evt][t2_W_lv_mask][0].item())
                    else: 
                        t2_l_data.append(genmatched_index[evt][t2_W_lv_mask][1].item()+14 if genmatched_index[evt][t2_W_lv_mask][0].item() >= 0 else genmatched_index[evt][t2_W_lv_mask][0].item())
                        t2_v_data.append(genmatched_index[evt][t2_W_lv_mask][0].item())
                else:
                    continue

                # data for "Source". 
                # # Merge jet, els, mus for each event
                merged_pt = np.concatenate( (jets_data[evt,:,0], els_data[evt,:,0], mus_data[evt,:,0] ), axis=0 )
                merged_eta = np.concatenate( (jets_data[evt,:,1], els_data[evt,:,1], mus_data[evt,:,1] ), axis=0 )
                merged_phi = np.concatenate( (jets_data[evt,:,2], els_data[evt,:,2], mus_data[evt,:,2] ), axis=0 )
                merged_mass = np.concatenate( (jets_data[evt,:,3], els_data[evt,:,3], mus_data[evt,:,3] ), axis=0 )
                merged_btag = np.concatenate( (jets_data[evt,:,4], np.full(len(els_data[evt,:,0]),-1), np.full(len(els_data[evt,:,0]),-1) ), axis=0 )

                # # collect the event into jet_xx_data (list)
                jets_pt_data.append(merged_pt)
                jets_eta_data.append(merged_eta)
                jets_phi_data.append(merged_phi)
                jets_mass_data.append(merged_mass)
                jets_btag_data.append(merged_btag)
                mask_data.append(np.full((len(merged_pt)), True, dtype='|b1'))



                count+=1



        source_group.create_dataset("pt", data=np.array(jets_pt_data).astype('<f4'))
        source_group.create_dataset("eta", data=np.array(jets_eta_data).astype('<f4'))
        source_group.create_dataset("phi", data=np.array(jets_phi_data).astype('<f4'))
        source_group.create_dataset("mass", data=np.array(jets_mass_data).astype('<f4'))
        source_group.create_dataset("btag", data=np.array(jets_btag_data).astype('<f4'))
        source_group.create_dataset("MASK", data=np.array(mask_data), dtype='|b1')

        t1_group.create_dataset("b", data=np.array(t1_b_data).astype('<i8'))
        t1_group.create_dataset("l", data=np.array(t1_l_data).astype('<i8'))
        t1_group.create_dataset("v", data=np.array(t1_v_data).astype('<i8'))
        t2_group.create_dataset("b", data=np.array(t2_b_data).astype('<i8'))
        t2_group.create_dataset("l", data=np.array(t2_l_data).astype('<i8'))
        t2_group.create_dataset("v", data=np.array(t2_v_data).astype('<i8'))

        print(count)

# "TT1L": {"diagram": {"t1": {"b": None, "W": {"q1": None, "q2": None, "SYMMETRY": ["q1", "q2"]}}, "t2": {"b": None, "W": {"l": None, "v": None}}}}
def convert_TT1L(yulei_file_path, out_file_path):

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
        
        # Add datasets for TARGETS/t1, t2
        # # Get data from infile["genpart"]: ["genpart_pt", "genpart_eta", "genpart_phi", "genpart_m", "genpart_index", "genpart_M1", "genpart_M2", "genpart_PID", "genpart_Status", "genmatched_index"]
        genpart_data = infile['genpart'][:]
        print("genpart_data dataset:", genpart_data.shape) # (10000, 12, 10)

        genpart_index = genpart_data[:,:,4]
        genpart_M1 = genpart_data[:,:,5]
        genpart_PID = genpart_data[:,:,7]
        genmatched_index = genpart_data[:,:,-1]

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
        t2_l_data = []
        t2_v_data = []
        count = 0

        # loop over each event
        for evt in range(n_evt):

            # check double assignment
            if has_double_assignment(genmatched_index[evt]):
                continue

            # get b mother pid
            b_genpart_M1 = genpart_M1[evt][(abs(genpart_PID[evt])==5)] # get b mother index 
            b_mother_mask = (genpart_index[evt]==b_genpart_M1[0]) | (genpart_index[evt]==b_genpart_M1[1]) # find location of b_mother in the array
            b_mother_pid = genpart_PID[evt][b_mother_mask]



            # check if both b are from top, if not reject event. pid of top is 6
            if (abs(b_mother_pid[0]) == 6) and (abs(b_mother_pid[1]) == 6):

                # genpart_index for b
                b_genpart_indices = genpart_index[evt][(abs(genpart_PID[evt])==5)]
                t1_b_index = b_genpart_indices[0]
                t2_b_index = b_genpart_indices[1]



                # find t1_child1 and t1_child2, they should come from a W with W_M1==b_genpart_M1[0]
                t1_idx = b_genpart_M1[0]
                mom_is_t1_mask = genpart_M1[evt]==t1_idx
                # get index of t1_W (both W and b have same mom t1)
                if genpart_index[evt][mom_is_t1_mask][0]==t1_b_index:
                    t1_W_idx = genpart_index[evt][mom_is_t1_mask][1]
                else :
                    t1_W_idx = genpart_index[evt][mom_is_t1_mask][0]
                # mask to find parts whose mother is t1_W, this should give us t1_child1 and t1_child2
                t1_W_child_mask = (genpart_M1[evt]==t1_W_idx)



                # find t2_child1 and t2_child2, they should come from a W with W_M1==b_genpart_M1[1]
                t2_idx = b_genpart_M1[1]
                mom_is_t2_mask = genpart_M1[evt]==t2_idx
                # get index of t2_W (both W and b have same mom t2)
                if genpart_index[evt][mom_is_t2_mask][0]==t2_b_index:
                    t2_W_idx = genpart_index[evt][mom_is_t2_mask][1]
                else :
                    t2_W_idx = genpart_index[evt][mom_is_t2_mask][0]
                # mask to find parts whose mother is t1_W, this should give us t2_child1 and t2_child2
                t2_W_child_mask = (genpart_M1[evt]==t2_W_idx)



                # start filling data
                # check if each evt has 2 parts whose mother is t1_W and 2 for t2_W, else reject evt
                if ( (len(genmatched_index[evt][t1_W_child_mask]) == 2) and (len(genmatched_index[evt][t2_W_child_mask]) == 2) ):

                    # distinguish hadronic or leptonic for t1 (determine t1 decay, then t2 is the other channel)
                    t1_children_pid = genpart_PID[evt][t1_W_child_mask]
                    print(t1_children_pid)
                    # quarks_pid:1~8, leptons_pid:11~18
                    if ( abs(t1_children_pid[0])<10 and abs(t1_children_pid[1])<10 ): # t1 is hadronic

                        # fill the t1 data
                        t1_b_data.append(genmatched_index[evt][genpart_index[evt]==t1_b_index].item())
                        t1_q1_data.append(genmatched_index[evt][t1_W_child_mask][0].item())
                        t1_q2_data.append(genmatched_index[evt][t1_W_child_mask][1].item())
                        
                        # fill in t2 data
                        t2_b_data.append(genmatched_index[evt][genpart_index[evt]==t2_b_index].item())
                        # distinguish t2_l and t2_v
                        if abs(genpart_PID[evt][t2_W_child_mask][0])== 11 or 13:
                            t2_l_data.append(genmatched_index[evt][t2_W_child_mask][0].item())
                            t2_v_data.append(genmatched_index[evt][t2_W_child_mask][1].item())
                        else:
                            t2_l_data.append(genmatched_index[evt][t2_W_child_mask][1].item())
                            t2_v_data.append(genmatched_index[evt][t2_W_child_mask][0].item())

                    else: # t1 is leptonic
                        # use t2 to fill the final t1 oultput data
                        t1_b_data.append(genmatched_index[evt][genpart_index[evt]==t2_b_index].item())
                        t1_q1_data.append(genmatched_index[evt][t2_W_child_mask][0].item())
                        t1_q2_data.append(genmatched_index[evt][t2_W_child_mask][1].item())
                        
                        # use t1 to fill the final t2 output data
                        t2_b_data.append(genmatched_index[evt][genpart_index[evt]==t1_b_index].item())
                        # distinguish t2_l and t2_v
                        if abs(genpart_PID[evt][t1_W_child_mask][0])== 11 or 13:
                            t2_l_data.append(genmatched_index[evt][t1_W_child_mask][0].item())
                            t2_v_data.append(genmatched_index[evt][t1_W_child_mask][1].item())
                        else:
                            t2_l_data.append(genmatched_index[evt][t1_W_child_mask][1].item())
                            t2_v_data.append(genmatched_index[evt][t1_W_child_mask][0].item())
                
                else:
                    continue



                # data for "Source" 
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
        t2_group.create_dataset("l", data=np.array(t2_l_data).astype('<i8'))
        t2_group.create_dataset("v", data=np.array(t2_v_data).astype('<i8'))

        print(count)
                
# "WJetsToLNu": {"diagram": {"W": {"l":None, "v": None}}}
def convert_WJetsToLNu(yulei_file_path, out_file_path):
    # Open yulei generated file and my output file
    with h5py.File(yulei_file_path, 'r') as infile, h5py.File(out_file_path, 'w') as outfile:
        
        # Create new groups for INPUTS
        inputs_group = outfile.create_group("INPUTS")
        source_group = inputs_group.create_group("Source")
        
        # Create new groups for TARGETS
        targets_group = outfile.create_group("TARGETS")
        l_group = targets_group.create_group("l")
        v_group = targets_group.create_group("v")
        
        # # Add datasets for INPUTS/Source 
        # # # Get data from infile[jets]: ["jet_pt","jet_eta","jet_phi","jet_m","jet_btag","jet_npart","jet_flavor"]
        jets_data = infile['jets'][:]
        print("jets dataset:", jets_data.shape) # (10000, 4, 7)

        # # MASK: if the jets is padded or not. (You may check, but I remember True means this jet is physical and False means this jet is padded one)
        # mask_data = np.full((10000, 4), True, dtype='|b1')
        # source_group.create_dataset("MASK", data=mask_data, dtype='|b1')

        
        # Add datasets for TARGETS/l, v
        # # Get data from infile["genpart"]: ["genpart_pt", "genpart_eta", "genpart_phi", "genpart_m", "genpart_index", "genpart_M1", "genpart_M2", "genpart_PID", "genpart_Status", "genmatched_index"]
        genpart_data = infile['genpart'][:]
        print("genpart_data dataset:", genpart_data.shape) # (10000, 12, 10)

        genpart_index = genpart_data[:,:,4]
        genpart_M1 = genpart_data[:,:,5]
        genpart_PID = genpart_data[:,:,7]
        genmatched_index = genpart_data[:,:,-1]

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
        l_data = []
        v_data = []
        count = 0

        # loop over each event
        for evt in range(n_evt):

            # check double assignment
            if has_double_assignment(genmatched_index[evt]):
                continue



            # get W index
            W_idx = genpart_index[evt][(abs(genpart_PID[evt])==24)]

            # find W's children
            mom_is_W_mask = (genpart_M1[evt]==W_idx)
            W_children_pid = genpart_PID[evt][mom_is_W_mask]
            W_children_genmatched_idx = genmatched_index[evt][mom_is_W_mask]
            # # if W doesn't have two children, reject event
            if len(W_children_pid)<2:
                continue
            # # distinguish l, v (bigger is v, smaller is l)
            if abs(W_children_pid[0]) > abs(W_children_pid[1]):
                v_data.append(W_children_genmatched_idx[0])
                l_data.append(W_children_genmatched_idx[1])
            else:
                v_data.append(W_children_genmatched_idx[1])
                l_data.append(W_children_genmatched_idx[0])

            print(W_children_pid)

        

            # data for "Source" 
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

        l_group.create_dataset("l", data=np.array(l_data).astype('<i8'))
        v_group.create_dataset("v", data=np.array(v_data).astype('<i8'))

        print(count)





if __name__ == '__main__':

    # # TTH
    # TTH_yulei_file_path = '/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/run_yulei_2/TTHadronics_367772000.h5'
    # TTH_out_file_path = '/pscratch/sd/w/weipow/OmniNet_Data/TTHadronics_367772000_omninet_10jets.h5'    
    # convert_TTHadronics(TTH_yulei_file_path, TTH_out_file_path)

    # # TT1L
    # TT1L_yulei_file_path = '/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/run_yulei_2/TT1L_367772000.h5'
    # TT1L_out_file_path = '/pscratch/sd/w/weipow/OmniNet_Data/TT1L_367772000_omninet_10jets.h5' 
    # convert_TT1L(TT1L_yulei_file_path, TT1L_out_file_path)

    # TT2L: 
    TT2L_yulei_file_path = '/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/run_yulei_2/TT2L_367772000.h5'
    TT2L_out_file_path = '/pscratch/sd/w/weipow/OmniNet_Data/TT2L_367772000_omninet_10jets.h5' 
    convert_TT2L(TT2L_yulei_file_path, TT2L_out_file_path)

    # WJetsToLNu: 
    # WJetsToLNu_yulei_file_path = '/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/run_yulei_2/WJetsToLNu_367772000.h5'
    # WJetsToLNu_out_file_path = '/pscratch/sd/w/weipow/OmniNet_Data/WJetsToLNu_367772000_omninet_10jets.h5' 
    # convert_WJetsToLNu(WJetsToLNu_yulei_file_path, WJetsToLNu_out_file_path)
