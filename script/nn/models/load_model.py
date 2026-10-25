from script.nn.models.GCLSTM import RecurrentGCN
from script.nn.models.ROLAND_DGNN import ROLANDGNN
from script.utils.util import logger
from script.nn.models.HTGN.HTGN import HTGN


# from script.models.static_baselines import VGAENet, GCNNet


def load_model(args):
    if args.model == 'HTGN':
        model = HTGN(args)
    if args.model == 'ROLAND':
        model_dim = {"input_dim": 33190, "hidden_conv_1": 16, "hidden_conv_2": 16}
        model = ROLANDGNN(model_dim, args.num_nodes)
    if args.model == 'GCLSTM':
        model = RecurrentGCN(args.nfeat, args.nhid)

    logger.info('using models {} '.format(args.model))
    return model
