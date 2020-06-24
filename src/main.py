from argparse import ArgumentParser

DEVICES = ['CPU', 'GPU', 'FPGA', 'MYRAID', 'HETERO']

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    general = parser.add_argument_group('General') 
    general.add_argument('-i', '--input', metavar='PATH', default='0',
                                  help="(optional) Path to the input video " \
                                      "('0' for camera, default)")