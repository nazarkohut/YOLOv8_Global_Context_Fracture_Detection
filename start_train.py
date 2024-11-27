import argparse
from ultralytics import YOLO

def parse_args():
    parse = argparse.ArgumentParser(description='Data Postprocess')
    parse.add_argument('--model', type=str, default=None, help='load the model')
    parse.add_argument('--pretained', type=str, default=None, help='pretained model')
    parse.add_argument('--data_dir', type=str, default=None, help='the dir to data')
    parse.add_argument('--device', type=str, default="0", help='device on which training will be held')
    parse.add_argument('--epochs', type=str, default="1", help='number of training epochs')
    args = parse.parse_args()
    return args

def main():
    args = parse_args()
    model = YOLO(args.model)#.load(args.pretained)
    model.train(data=args.data_dir, device=args.device)

if __name__ == '__main__':
    main()
