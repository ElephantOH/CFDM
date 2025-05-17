import argparse
import ast

def parse_dict(value):
    try:
        return ast.literal_eval(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")
def load_config(parser):
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--wait_time', type=float, default=0 * 60 * 60)
    parser.add_argument('--gpu_chose', type=int, default=0)
    parser.add_argument('--use_model_name', type=str, default='FDM')

    # model
    parser.add_argument('--num_channels', type=int, default=64, help='number of initial channels in denosing model')
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--conditional_type', type=str, default='coord', help='model time or coord')
    parser.add_argument('--z_emb_dim', type=int, default=100)
    parser.add_argument('--z_emb_channels', nargs='+', type=int, default=[256, 256, 256, 256])
    parser.add_argument('--t_emb_dim', type=int, default=64)
    parser.add_argument('--t_emb_channels', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--level_channels', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
    parser.add_argument('--attn_levels', default=(16,))
    parser.add_argument('--use_cross_attn', action='store_true', default=False)
    parser.add_argument('--num_resblocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='type of resnet block, choice in biggan and ddpm')
    parser.add_argument('--use_tanh_final', action='store_false', default=True)
    parser.add_argument('--phase', type=str, default='train', help='model train_fdm, train_cfdm or test_cfdm')
    parser.add_argument('--output_complete', action='store_true', default=True)
    parser.add_argument('--use_multi_flow', action='store_true', default=True)
    parser.add_argument('--network_type', default='normal', help='choose of normal, large, max')

    # training
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='learning rate')
    parser.add_argument('--lrf', type=float, default=1e-5, help='learning rate final')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=True, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--save_content', action='store_true', default=True)
    parser.add_argument('--save_content_every', type=int, default=5, help='save content')
    parser.add_argument('--save_ckpt_every', type=int, default=5, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1', type=float, default=1, help='weightening of loss')
    parser.add_argument('--lambda_l2', type=float, default=1, help='weightening of loss')
    parser.add_argument('--lambda_perceptual', type=float, default=1, help='weightening of loss')
    parser.add_argument('--log_iteration', type=int, default=10)
    parser.add_argument('--use_reg', action='store_true', default=False)

    # val
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--val_every', type=int, default=5, help='validation every x epochs')
    parser.add_argument('--val_batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--which_epoch', type=int, default=120)
    parser.add_argument('--sample_fixed', action='store_true', default=False)

    # ddp
    parser.add_argument('--ddp', action='store_true', default=True)
    parser.add_argument('--num_proc_node', type=int, default=4, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
    parser.add_argument('--port_num', type=str, default='6023', help='port selection for code')

    ## train_fdm

    ## train alpha beta
    parser.add_argument('--train_alpha', nargs='+', type=float, default=[0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument('--train_beta', nargs='+', type=float, default=[1.0, 1.0, 1.0, 1.0, 1.0])
    ## noise gate
    parser.add_argument('--dynamics_noise_gate', action='store_true', default=False)
    parser.add_argument('--dng_id', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--dynamics_noise_gate_epoch', type=int, default=10)
    parser.add_argument('--max_dynamic_threshold', type=int, default=0.008)
    parser.add_argument('--ideal_noise_gate', nargs='+', type=float, default=[1.0, 1.0, 1.0, 1.0, 1.0])
    ## test
    # parser.add_argument('--test_alpha', nargs='+', type=float, default=[0.0, 0.27569055543639487, 0.3478207454179226, 0.38878689145710116])
    # parser.add_argument('--test_beta', nargs='+', type=float, default=[1.0, 0.9709511995315552, 0.7927531003952026, 0.36576129496097565])
    ## test zero
    parser.add_argument('--test_alpha', nargs='+', type=float, default=[0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument('--test_beta', nargs='+', type=float, default=[1.0, 1.0, 1.0, 1.0, 1.0])
    ## tfw
    # parser.add_argument('--test_alpha', nargs='+', type=float, default=[1.0, 0.590149462223053, 0.7517210245132446, 0.5703374147415161])
    # parser.add_argument('--test_beta', nargs='+', type=float, default=[0.0, 0.4881748557090759, -0.028303980827331543, 1.2943629026412964])
    ## llvip
    # parser.add_argument('--test_alpha', nargs='+', type=float, default=[1.0, 0.590149462223053, 0.7517210245132446, 0.5703374147415161])
    # parser.add_argument('--test_beta', nargs='+', type=float, default=[0.0, 0.4881748557090759, -0.028303980827331543, 1.2943629026412964])
    ## brats 1to2
    # parser.add_argument('--test_alpha', nargs='+', type=float, default=[1.0, 1.038264513015747, 0.3693735599517822, -0.34250444173812866])
    # parser.add_argument('--test_beta', nargs='+', type=float, default=[0.0, 0.6858664155006409, 0.39165475964546204, 0.4291553199291229])
    ## oasis 2to1
    # parser.add_argument('--test_alpha', nargs='+', type=float, default=[0.9722041487693787, 0.850620448589325, 0.6705297827720642, 0.0701717659831047])
    # parser.add_argument('--test_beta', nargs='+', type=float, default=[0.02469334565103054, 0.4064328372478485, 0.11989399790763855, 0.001114287762902677])

    parser.add_argument('--image_size', type=int, default=256, help='size of image')
    parser.add_argument('--cfdm_batch_size', type=int, default=2)
    parser.add_argument('--cfdm_dataset_size', type=int, default=100)
    parser.add_argument('--cfdm_lr', type=float, default=1e-3)
    parser.add_argument('--cfdm_lrf', type=float, default=1e-5)
    parser.add_argument('--input_channels', type=int, default=3, help='channel of image')
    parser.add_argument('--input_path', default='', help='path to input data')
    parser.add_argument('--checkpoint_path', default='', help='path to output saves')
    parser.add_argument('--normed', action='store_true', default=False)
    parser.add_argument('--source', type=str, default='T1', help='contrast selection for model')
    parser.add_argument('--target', type=str, default='T2', help='contrast selection for model')
    parser.add_argument('--fractal_num', type=int, default=4)
    parser.add_argument('--fractal_init_state', type=str, default='source')
    parser.add_argument('--padding', action='store_true', default=False)
    parser.add_argument('--fractal_sizes', nargs='+', type=int, default=
        [256,
         256,
         128,
         64
         ]
    )
    parser.add_argument('--fractal_padding', nargs='+', type=int, default=
        [0,
         0,
         0,
         0
         ]
    )
    parser.add_argument('--fractal_channels', nargs='+', type=int, default=
        [ 64,
          64,
          32,
          16
         ]
    )
    parser.add_argument('--fractal_levels', nargs='+', type=int, default=
        [[1, 1, 2, 2, 4, 4],
         [1, 1, 2, 2, 4],
         [1, 1, 2],
         [1, 1],
        ]
    )
    parser.add_argument('--fractal_attns', nargs='+', type=int, default=
        [(16,),
         (16,),
         (0,),
         (0,),
         ]
    )
    parser.add_argument('--fractal_lr', nargs='+', type=int, default=
        [1.5e-4,
         1.5e-4,
         1e-4,
         1e-5
         ]
     )
    parser.add_argument('--fractal_lrf', nargs='+', type=int, default=
        [1e-5,
         1e-5,
         5e-6,
         1e-7
         ]
    )

    # fi
    parser.add_argument('--frequency_i', action='store_true', default=False)
    parser.add_argument('--frequency_type', type=str, default='haar', help='frequency type: fft, haar')
    parser.add_argument('--num_bands', type=int, default=4)

    return parser
