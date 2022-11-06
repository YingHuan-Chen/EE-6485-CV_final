import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--frame_path', type=str, default='data/YouTube_Highlights_processed',
                        help='path to the image')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
                        
    return parser.parse_args()