import torch
import logging
import src
import src.model
CHECKPOINT_DIR = "./checkpoints/"

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the model test...")
    args = src.model.ModelArgs()
    model = src.model.Model(args)
    dummy_input = torch.randn(1, 10, args.input_size)
    output = model(dummy_input)
    print("Model output shape:", output.shape)
    print("Model output:", output)
    logging.info("Finished model test.")

if __name__ == "__main__":
    main()