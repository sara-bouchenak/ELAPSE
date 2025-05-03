from .adaptivedataloader import AdaptiveDSSDataLoader
from cords.selectionstrategies.SL import GLISTERStrategy
import time
import copy
import pandas as pd
import torch
import os

class GLISTERDataLoader(AdaptiveDSSDataLoader):
    def __init__(self, train_loader, val_loader, dss_args, logger, dataset_name=None, model_name=None, ratio=None, *args, **kwargs):
        """
        Constructor function

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            Training dataset loader
        val_loader: torch.utils.data.DataLoader
            Validation dataset loader
        dss_args: dict
            Data subset selection arguments
        logger: Logger
            Logger instance for debugging and information
        dataset_name: str, optional
            Name of the dataset for conditional logic
        """
        self.dataset_name = dataset_name  
        self.model_name = model_name  
        self.ratio = ratio
        super(GLISTERDataLoader, self).__init__(train_loader, val_loader, dss_args, logger, *args, **kwargs)

        # Initialize GLISTER strategy
        self.strategy = GLISTERStrategy(
            train_loader, val_loader, copy.deepcopy(dss_args['model']),
            dss_args['loss'], dss_args['eta'], dss_args['device'],
            dss_args['num_classes'], dss_args['linear_layer'],
            dss_args['selection_type'], dss_args['greedy'], logger,
            r=dss_args.get('r', 0)
        )
        self.train_model = dss_args['model']
        self.logger.debug('GLISTER dataloader initialized.')

    def _resample_subset_indices(self):
        """
        Function that calls the GLISTER subset selection strategy to sample new subset indices and weights.
        """
        start = time.time()
        self.logger.debug('Epoch: {0:d}, requires subset selection. '.format(self.cur_epoch))
        cached_state_dict = copy.deepcopy(self.train_model.state_dict())
        clone_dict = copy.deepcopy(self.train_model.state_dict())
        subset_indices, subset_weights = self.strategy.select(self.budget, clone_dict)
        self.train_model.load_state_dict(cached_state_dict)
        end = time.time()
        self.logger.info('Epoch: {0:d}, GLISTER dataloader subset selection finished, takes {1:.4f}. '.format(self.cur_epoch, (end - start)))

        if self.dataset_name == 'ars':
            try:
                # Extract data based on subset indices
                subset_data = [self.train_loader.dataset[idx] for idx in subset_indices]
                subset_features = [data[0].numpy() for data in subset_data]
                subset_labels = [data[1].item() for data in subset_data]
                sensitive_attributes = [data[2].numpy() if isinstance(data[2], torch.Tensor) else data[2] for data in subset_data]

                # Convert features and attributes to DataFrame
                df_features = pd.DataFrame(subset_features, columns=['x-axis', 'y-axis', 'z-axis', 'sensor', 'rssi', 'phase', 'frequency', 'gender', 'room_id'])
                sensitive_attributes_df = pd.DataFrame(sensitive_attributes, columns=['gender'])
                #df = pd.concat([df_features, sensitive_attributes_df], axis=1)
                df = pd.concat([df_features], axis=1)
                df['activity'] = subset_labels

                # Save to CSV
                os.makedirs("../data/" +str(self.dataset_name)+"/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/", exist_ok=True)
                csv_filename = f"../data/" +str(self.dataset_name)+"/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/selected_data_"+ str(self.model_name)+"_epoch_"+str(self.cur_epoch)+".csv"
                df.to_csv(csv_filename, index=False)
            except Exception as e:
                self.logger.error(f"Error during data processing: {e}")

        
        # Adult
        
        if self.dataset_name == 'census':
            try:
                # Extract data based on subset indices
                subset_data = [self.train_loader.dataset[idx] for idx in subset_indices]
                subset_features = [data[0].numpy() for data in subset_data]
                subset_labels = [data[1].item() for data in subset_data]
                sensitive_attributes = [data[2].numpy() if isinstance(data[2], torch.Tensor) else data[2] for data in subset_data]

                # Convert features and attributes to DataFrame
                df_features = pd.DataFrame(subset_features, columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country'])
                sensitive_attributes_df = pd.DataFrame(sensitive_attributes, columns=['sex', 'age', 'race'])
                df = pd.concat([df_features], axis=1)
                df['income'] = subset_labels

                # Save to CSV
                os.makedirs("../data/adult/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/", exist_ok=True)
                csv_filename = f"../data/adult/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/selected_data_"+ str(self.model_name)+"_epoch_"+str(self.cur_epoch)+".csv"
                df.to_csv(csv_filename, index=False)
            except Exception as e:
                self.logger.error(f"Error during data processing: {e}")

        # KDD
        
        if self.dataset_name == 'kdd':
            try:
                # Extract data based on subset indices
                subset_data = [self.train_loader.dataset[idx] for idx in subset_indices]
                subset_features = [data[0].numpy() for data in subset_data]
                subset_labels = [data[1].item() for data in subset_data]
                sensitive_attributes = [data[2].numpy() if isinstance(data[2], torch.Tensor) else data[2] for data in subset_data]

                # Convert features and attributes to DataFrame
                df_features = pd.DataFrame(subset_features, columns=[ 'class_worker_Category_Federal government', 'class_worker_Category_Local government', 'class_worker_Category_Never worked', 'class_worker_Category_Not in universe', 'class_worker_Category_Private', 'class_worker_Category_Self-employed-incorporated', 'class_worker_Category_Self-employed-not incorporated', 'class_worker_Category_State government', 'class_worker_Category_Without pay', 'education_Category_10th grade', 'education_Category_11th grade', 'education_Category_12th grade no diploma', 'education_Category_1st 2nd 3rd or 4th grade', 'education_Category_5th or 6th grade', 'education_Category_7th and 8th grade', 'education_Category_9th grade', 'education_Category_Associates degree-academic program', 'education_Category_Associates degree-occup /vocational', 'education_Category_Bachelors degree(BA AB BS)', 'education_Category_Children', 'education_Category_Doctorate degree(PhD EdD)', 'education_Category_High school graduate', 'education_Category_Less than 1st grade', 'education_Category_Masters degree(MA MS MEng MEd MSW MBA)', 'education_Category_Prof school degree (MD DDS DVM LLB JD)', 'education_Category_Some college but no degree', 'hs_college_Category_College or university', 'hs_college_Category_High school', 'hs_college_Category_Not in universe', 'marital_stat_Category_Divorced', 'marital_stat_Category_Married-A F spouse present', 'marital_stat_Category_Married-civilian spouse present', 'marital_stat_Category_Married-spouse absent', 'marital_stat_Category_Never married', 'marital_stat_Category_Separated', 'marital_stat_Category_Widowed', 'major_ind_code_Category_Agriculture', 'major_ind_code_Category_Armed Forces', 'major_ind_code_Category_Business and repair services', 'major_ind_code_Category_Communications', 'major_ind_code_Category_Construction', 'major_ind_code_Category_Education', 'major_ind_code_Category_Entertainment', 'major_ind_code_Category_Finance insurance and real estate', 'major_ind_code_Category_Forestry and fisheries', 'major_ind_code_Category_Hospital services', 'major_ind_code_Category_Manufacturing-durable goods', 'major_ind_code_Category_Manufacturing-nondurable goods', 'major_ind_code_Category_Medical except hospital', 'major_ind_code_Category_Mining', 'major_ind_code_Category_Not in universe or children', 'major_ind_code_Category_Other professional services', 'major_ind_code_Category_Personal services except private HH', 'major_ind_code_Category_Private household services', 'major_ind_code_Category_Public administration', 'major_ind_code_Category_Retail trade', 'major_ind_code_Category_Social services', 'major_ind_code_Category_Transportation', 'major_ind_code_Category_Utilities and sanitary services', 'major_ind_code_Category_Wholesale trade', 'major_occ_code_Category_Adm support including clerical', 'major_occ_code_Category_Armed Forces', 'major_occ_code_Category_Executive admin and managerial', 'major_occ_code_Category_Farming forestry and fishing', 'major_occ_code_Category_Handlers equip cleaners etc ', 'major_occ_code_Category_Machine operators assmblrs & inspctrs', 'major_occ_code_Category_Not in universe', 'major_occ_code_Category_Other service', 'major_occ_code_Category_Precision production craft & repair', 'major_occ_code_Category_Private household services', 'major_occ_code_Category_Professional specialty', 'major_occ_code_Category_Protective services', 'major_occ_code_Category_Sales', 'major_occ_code_Category_Technicians and related support', 'major_occ_code_Category_Transportation and material moving', 'union_member_Category_No', 'union_member_Category_Not in universe', 'union_member_Category_Yes', 'unemp_reason_Category_Job leaver', 'unemp_reason_Category_Job loser - on layoff', 'unemp_reason_Category_New entrant', 'unemp_reason_Category_Not in universe', 'unemp_reason_Category_Other job loser', 'unemp_reason_Category_Re-entrant', 'full_or_part_emp_Category_Children or Armed Forces', 'full_or_part_emp_Category_Full-time schedules', 'full_or_part_emp_Category_Not in labor force', 'full_or_part_emp_Category_PT for econ reasons usually FT', 'full_or_part_emp_Category_PT for econ reasons usually PT', 'full_or_part_emp_Category_PT for non-econ reasons usually FT', 'full_or_part_emp_Category_Unemployed full-time', 'full_or_part_emp_Category_Unemployed part- time', 'tax_filer_stat_Category_Head of household', 'tax_filer_stat_Category_Joint both 65+', 'tax_filer_stat_Category_Joint both under 65', 'tax_filer_stat_Category_Joint one under 65 & one 65+', 'tax_filer_stat_Category_Nonfiler', 'tax_filer_stat_Category_Single', 'region_prev_res_Category_Abroad', 'region_prev_res_Category_Midwest', 'region_prev_res_Category_Northeast', 'region_prev_res_Category_Not in universe', 'region_prev_res_Category_South', 'region_prev_res_Category_West', 'state_prev_res_Category_Abroad', 'state_prev_res_Category_Alabama', 'state_prev_res_Category_Alaska', 'state_prev_res_Category_Arizona', 'state_prev_res_Category_Arkansas', 'state_prev_res_Category_California', 'state_prev_res_Category_Colorado', 'state_prev_res_Category_Connecticut', 'state_prev_res_Category_Delaware', 'state_prev_res_Category_District of Columbia', 'state_prev_res_Category_Florida', 'state_prev_res_Category_Georgia', 'state_prev_res_Category_Idaho', 'state_prev_res_Category_Illinois', 'state_prev_res_Category_Indiana', 'state_prev_res_Category_Iowa', 'state_prev_res_Category_Kansas', 'state_prev_res_Category_Kentucky', 'state_prev_res_Category_Louisiana', 'state_prev_res_Category_Maine', 'state_prev_res_Category_Maryland', 'state_prev_res_Category_Massachusetts', 'state_prev_res_Category_Michigan', 'state_prev_res_Category_Minnesota', 'state_prev_res_Category_Mississippi', 'state_prev_res_Category_Missouri', 'state_prev_res_Category_Montana', 'state_prev_res_Category_Nebraska', 'state_prev_res_Category_Nevada', 'state_prev_res_Category_New Hampshire', 'state_prev_res_Category_New Jersey', 'state_prev_res_Category_New Mexico', 'state_prev_res_Category_New York', 'state_prev_res_Category_North Carolina', 'state_prev_res_Category_North Dakota', 'state_prev_res_Category_Not in universe', 'state_prev_res_Category_Ohio', 'state_prev_res_Category_Oklahoma', 'state_prev_res_Category_Oregon', 'state_prev_res_Category_Pennsylvania', 'state_prev_res_Category_South Carolina', 'state_prev_res_Category_South Dakota', 'state_prev_res_Category_Tennessee', 'state_prev_res_Category_Texas', 'state_prev_res_Category_Utah', 'state_prev_res_Category_Vermont', 'state_prev_res_Category_Virginia', 'state_prev_res_Category_West Virginia', 'state_prev_res_Category_Wisconsin', 'state_prev_res_Category_Wyoming', 'det_hh_fam_stat_Category_Child 18+ ever marr Not in a subfamily', 'det_hh_fam_stat_Category_Child 18+ ever marr RP of subfamily', 'det_hh_fam_stat_Category_Child 18+ never marr Not in a subfamily', 'det_hh_fam_stat_Category_Child 18+ never marr RP of subfamily', 'det_hh_fam_stat_Category_Child 18+ spouse of subfamily RP', 'det_hh_fam_stat_Category_Child <18 ever marr RP of subfamily', 'det_hh_fam_stat_Category_Child <18 ever marr not in subfamily', 'det_hh_fam_stat_Category_Child <18 never marr RP of subfamily', 'det_hh_fam_stat_Category_Child <18 never marr not in subfamily', 'det_hh_fam_stat_Category_Child <18 spouse of subfamily RP', 'det_hh_fam_stat_Category_Child under 18 of RP of unrel subfamily', 'det_hh_fam_stat_Category_Grandchild 18+ ever marr RP of subfamily', 'det_hh_fam_stat_Category_Grandchild 18+ ever marr not in subfamily', 'det_hh_fam_stat_Category_Grandchild 18+ never marr RP of subfamily', 'det_hh_fam_stat_Category_Grandchild 18+ never marr not in subfamily', 'det_hh_fam_stat_Category_Grandchild 18+ spouse of subfamily RP', 'det_hh_fam_stat_Category_Grandchild <18 ever marr not in subfamily', 'det_hh_fam_stat_Category_Grandchild <18 never marr RP of subfamily', 'det_hh_fam_stat_Category_Grandchild <18 never marr child of subfamily RP', 'det_hh_fam_stat_Category_Grandchild <18 never marr not in subfamily', 'det_hh_fam_stat_Category_Householder', 'det_hh_fam_stat_Category_In group quarters', 'det_hh_fam_stat_Category_Nonfamily householder', 'det_hh_fam_stat_Category_Other Rel 18+ ever marr RP of subfamily', 'det_hh_fam_stat_Category_Other Rel 18+ ever marr not in subfamily', 'det_hh_fam_stat_Category_Other Rel 18+ never marr RP of subfamily', 'det_hh_fam_stat_Category_Other Rel 18+ never marr not in subfamily', 'det_hh_fam_stat_Category_Other Rel 18+ spouse of subfamily RP', 'det_hh_fam_stat_Category_Other Rel <18 ever marr RP of subfamily', 'det_hh_fam_stat_Category_Other Rel <18 ever marr not in subfamily', 'det_hh_fam_stat_Category_Other Rel <18 never marr child of subfamily RP', 'det_hh_fam_stat_Category_Other Rel <18 never marr not in subfamily', 'det_hh_fam_stat_Category_Other Rel <18 never married RP of subfamily', 'det_hh_fam_stat_Category_Other Rel <18 spouse of subfamily RP', 'det_hh_fam_stat_Category_RP of unrelated subfamily', 'det_hh_fam_stat_Category_Secondary individual', 'det_hh_fam_stat_Category_Spouse of RP of unrelated subfamily', 'det_hh_fam_stat_Category_Spouse of householder', 'det_hh_summ_Category_Child 18 or older', 'det_hh_summ_Category_Child under 18 ever married', 'det_hh_summ_Category_Child under 18 never married', 'det_hh_summ_Category_Group Quarters- Secondary individual', 'det_hh_summ_Category_Householder', 'det_hh_summ_Category_Nonrelative of householder', 'det_hh_summ_Category_Other relative of householder', 'det_hh_summ_Category_Spouse of householder', 'mig_chg_msa_Category_Abroad to MSA', 'mig_chg_msa_Category_Abroad to nonMSA', 'mig_chg_msa_Category_MSA to MSA', 'mig_chg_msa_Category_MSA to nonMSA', 'mig_chg_msa_Category_NonMSA to MSA', 'mig_chg_msa_Category_NonMSA to nonMSA', 'mig_chg_msa_Category_Nonmover', 'mig_chg_msa_Category_Not identifiable', 'mig_chg_msa_Category_Not in universe', 'mig_chg_reg_Category_Abroad', 'mig_chg_reg_Category_Different county same state', 'mig_chg_reg_Category_Different division same region', 'mig_chg_reg_Category_Different region', 'mig_chg_reg_Category_Different state same division', 'mig_chg_reg_Category_Nonmover', 'mig_chg_reg_Category_Not in universe', 'mig_chg_reg_Category_Same county', 'mig_move_reg_Category_Abroad', 'mig_move_reg_Category_Different county same state', 'mig_move_reg_Category_Different state in Midwest', 'mig_move_reg_Category_Different state in Northeast', 'mig_move_reg_Category_Different state in South', 'mig_move_reg_Category_Different state in West', 'mig_move_reg_Category_Nonmover', 'mig_move_reg_Category_Not in universe', 'mig_move_reg_Category_Same county', 'mig_same_Category_No', 'mig_same_Category_Not in universe under 1 year old', 'mig_same_Category_Yes', 'mig_prev_sunbelt_Category_No', 'mig_prev_sunbelt_Category_Not in universe', 'mig_prev_sunbelt_Category_Yes', 'fam_under_18_Category_Both parents present', 'fam_under_18_Category_Father only present', 'fam_under_18_Category_Mother only present', 'fam_under_18_Category_Neither parent present', 'fam_under_18_Category_Not in universe', 'country_father_Category_Cambodia', 'country_father_Category_Canada', 'country_father_Category_China', 'country_father_Category_Columbia', 'country_father_Category_Cuba', 'country_father_Category_Dominican-Republic', 'country_father_Category_Ecuador', 'country_father_Category_El-Salvador', 'country_father_Category_England', 'country_father_Category_France', 'country_father_Category_Germany', 'country_father_Category_Greece', 'country_father_Category_Guatemala', 'country_father_Category_Haiti', 'country_father_Category_Holand-Netherlands', 'country_father_Category_Honduras', 'country_father_Category_Hong Kong', 'country_father_Category_Hungary', 'country_father_Category_India', 'country_father_Category_Iran', 'country_father_Category_Ireland', 'country_father_Category_Italy', 'country_father_Category_Jamaica', 'country_father_Category_Japan', 'country_father_Category_Laos', 'country_father_Category_Mexico', 'country_father_Category_Nicaragua', 'country_father_Category_Outlying-U S (Guam USVI etc)', 'country_father_Category_Panama', 'country_father_Category_Peru', 'country_father_Category_Philippines', 'country_father_Category_Poland', 'country_father_Category_Portugal', 'country_father_Category_Puerto-Rico', 'country_father_Category_Scotland', 'country_father_Category_South Korea', 'country_father_Category_Taiwan', 'country_father_Category_Thailand', 'country_father_Category_Trinadad&Tobago', 'country_father_Category_United-States', 'country_father_Category_Vietnam', 'country_father_Category_Yugoslavia', 'country_mother_Category_Cambodia', 'country_mother_Category_Canada', 'country_mother_Category_China', 'country_mother_Category_Columbia', 'country_mother_Category_Cuba', 'country_mother_Category_Dominican-Republic', 'country_mother_Category_Ecuador', 'country_mother_Category_El-Salvador', 'country_mother_Category_England', 'country_mother_Category_France', 'country_mother_Category_Germany', 'country_mother_Category_Greece', 'country_mother_Category_Guatemala', 'country_mother_Category_Haiti', 'country_mother_Category_Holand-Netherlands', 'country_mother_Category_Honduras', 'country_mother_Category_Hong Kong', 'country_mother_Category_Hungary', 'country_mother_Category_India', 'country_mother_Category_Iran', 'country_mother_Category_Ireland', 'country_mother_Category_Italy', 'country_mother_Category_Jamaica', 'country_mother_Category_Japan', 'country_mother_Category_Laos', 'country_mother_Category_Mexico', 'country_mother_Category_Nicaragua', 'country_mother_Category_Outlying-U S (Guam USVI etc)', 'country_mother_Category_Panama', 'country_mother_Category_Peru', 'country_mother_Category_Philippines', 'country_mother_Category_Poland', 'country_mother_Category_Portugal', 'country_mother_Category_Puerto-Rico', 'country_mother_Category_Scotland', 'country_mother_Category_South Korea', 'country_mother_Category_Taiwan', 'country_mother_Category_Thailand', 'country_mother_Category_Trinadad&Tobago', 'country_mother_Category_United-States', 'country_mother_Category_Vietnam', 'country_mother_Category_Yugoslavia', 'country_self_Category_Cambodia', 'country_self_Category_Canada', 'country_self_Category_China', 'country_self_Category_Columbia', 'country_self_Category_Cuba', 'country_self_Category_Dominican-Republic', 'country_self_Category_Ecuador', 'country_self_Category_El-Salvador', 'country_self_Category_England', 'country_self_Category_France', 'country_self_Category_Germany', 'country_self_Category_Greece', 'country_self_Category_Guatemala', 'country_self_Category_Haiti', 'country_self_Category_Holand-Netherlands', 'country_self_Category_Honduras', 'country_self_Category_Hong Kong', 'country_self_Category_Hungary', 'country_self_Category_India', 'country_self_Category_Iran', 'country_self_Category_Ireland', 'country_self_Category_Italy', 'country_self_Category_Jamaica', 'country_self_Category_Japan', 'country_self_Category_Laos', 'country_self_Category_Mexico', 'country_self_Category_Nicaragua', 'country_self_Category_Outlying-U S (Guam USVI etc)', 'country_self_Category_Panama', 'country_self_Category_Peru', 'country_self_Category_Philippines', 'country_self_Category_Poland', 'country_self_Category_Portugal', 'country_self_Category_Puerto-Rico', 'country_self_Category_Scotland', 'country_self_Category_South Korea', 'country_self_Category_Taiwan', 'country_self_Category_Thailand', 'country_self_Category_Trinadad&Tobago', 'country_self_Category_United-States', 'country_self_Category_Vietnam', 'country_self_Category_Yugoslavia', 'citizenship_Category_Foreign born- Not a citizen of U S ', 'citizenship_Category_Foreign born- U S citizen by naturalization', 'citizenship_Category_Native- Born abroad of American Parent(s)', 'citizenship_Category_Native- Born in Puerto Rico or U S Outlying', 'citizenship_Category_Native- Born in the United States', 'vet_question_Category_No', 'vet_question_Category_Not in universe', 'vet_question_Category_Yes', 'hisp_origin_Category_All other', 'hisp_origin_Category_Central or South American', 'hisp_origin_Category_Chicano', 'hisp_origin_Category_Cuban', 'hisp_origin_Category_Do not know', 'hisp_origin_Category_Mexican (Mexicano)', 'hisp_origin_Category_Mexican-American', 'hisp_origin_Category_Other Spanish', 'hisp_origin_Category_Puerto Rican', 'det_ind_code', 'det_occ_code', 'wage_per_hour', 'capital_gains', 'capital_losses', 'stock_dividends', 'unknown', 'num_emp', 'own_or_self', 'vet_benefits', 'weeks_worked', 'year', 'race', 'sex', 'age' ])
                sensitive_attributes_df = pd.DataFrame(sensitive_attributes, columns=['race','sex','age'])
                df = pd.concat([df_features], axis=1) 
                df['income'] = subset_labels

                # Save to CSV
                os.makedirs("../data/" +str(self.dataset_name)+"/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/", exist_ok=True)
                csv_filename = f"../data/" +str(self.dataset_name)+"/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/selected_data_"+ str(self.model_name)+"_epoch_"+str(self.cur_epoch)+".csv"
                df.to_csv(csv_filename, index=False)
            except Exception as e:
                self.logger.error(f"Error during data processing: {e}")

        if self.dataset_name == 'dc':
            try:
                # Extract data based on subset indices
                subset_data = [self.train_loader.dataset[idx] for idx in subset_indices]
                subset_features = [data[0].numpy() for data in subset_data]
                subset_labels = [data[1].item() for data in subset_data]
                sensitive_attributes = [data[2].numpy() if isinstance(data[2], torch.Tensor) else data[2] for data in subset_data]

                # Convert features and attributes to DataFrame
                df_features = pd.DataFrame(subset_features, columns=['household_position', 'household_size', 'prev_residence_place', 'citizenship', 'country_birth', 'edu_level', 'economic_status', 'cur_eco_activity', 'Marital_status', 'age', 'sex'])
                sensitive_attributes_df = pd.DataFrame(sensitive_attributes, columns=['age', 'sex'])
                df = pd.concat([df_features], axis=1)
                df['occupation'] = subset_labels

                # Save to CSV
                os.makedirs("../data/" +str(self.dataset_name)+"/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/", exist_ok=True)
                csv_filename = f"../data/" +str(self.dataset_name)+"/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/selected_data_"+ str(self.model_name)+"_epoch_"+str(self.cur_epoch)+".csv"
                df.to_csv(csv_filename, index=False)
            except Exception as e:
                self.logger.error(f"Error during data processing: {e}")


        if self.dataset_name == 'mobiact':
            try:
                # Extract data based on subset indices
                subset_data = [self.train_loader.dataset[idx] for idx in subset_indices]
                subset_features = [data[0].numpy() for data in subset_data]
                subset_labels = [data[1].item() for data in subset_data]
                sensitive_attributes = [data[2].numpy() if isinstance(data[2], torch.Tensor) else data[2] for data in subset_data]

                # Convert features and attributes to DataFrame
                df_features = pd.DataFrame(subset_features, columns=[ 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll','id', 'age', 'gender' ])
                sensitive_attributes_df = pd.DataFrame(sensitive_attributes, columns=['age', 'gender'])
                df = pd.concat([df_features], axis=1)
                df['activity'] = subset_labels

                # Save to CSV
                os.makedirs("../data/" +str(self.dataset_name)+"/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/", exist_ok=True)
                csv_filename = f"../data/" +str(self.dataset_name)+"/"+str(self.dataset_name)+"_"+str(self.ratio)+"/selected_data_glister/"+ str(self.model_name)+"/selected_data_"+ str(self.model_name)+"_epoch_"+str(self.cur_epoch)+".csv"
                df.to_csv(csv_filename, index=False)
            except Exception as e:
                self.logger.error(f"Error during data processing: {e}")


        return subset_indices, subset_weights
