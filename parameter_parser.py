import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--cuda', type=int, default=0, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--exp', type=str, default='Unlearn', choices=["Unlearn", "Attack"])
    parser.add_argument('--method', type=str, default='FairGU')
    parser.add_argument('--target_model', type=str, default='FGCN', choices=["SAGE", "GAT", 'MLP', "GCN", "GIN", "SGC", "FGCN", "FGAT"])
    parser.add_argument('--inductive', type=str, default='normal', choices=['cluster-gcn', 'graphsaint', 'normal'])
    parser.add_argument('--dataset_name', type=str, default='pokec_z', choices=["pokec_z", "pokec_n", "income", "credit", "cora", "citeseer"])
    parser.add_argument('--unlearn_ratio', type=float, default=0.05)

    ########################## training parameters ###########################
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--num_runs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2048)

    parser.add_argument('--alpha', type=float, default=100, help='covariance weight')
    parser.add_argument('--beta', type=float, default=1, help='adversarial weight')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units')
    parser.add_argument('--sens_number', type=int, default=200, help='Number of sensitive attributes')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval for training progress')
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--acc_threshold', type=float, default=0.68, help='Minimum ACC score threshold for model selection')
    parser.add_argument('--f1_threshold', type=float, default=0.68, help='Minimum F1 score threshold for model selection')

    parser.add_argument('--fim_lower_bound', type=float, default=1.0)
    parser.add_argument('--fim_exponent', type=float, default=1.0)
    parser.add_argument('--fim_dampening_constant', type=float, default=1.0)
    parser.add_argument('--fim_selection_weighting', type=float, default=10.0)

    parser.add_argument('--csv_name', type=str, default='result/FairGU_results')
    parser.add_argument('--attack_csv_name', type=str, default='result/attack_results')

    ########################## attack parameters ######################
    parser.add_argument('--attack_file_prefix', type=str, default='attack_materials/FairGU',
                        help='prefix for saved attack checkpoints')
    parser.add_argument('--attack_num_runs', type=int, default=1, help='repeat attack evaluation times')

    args = vars(parser.parse_args())

    return args
