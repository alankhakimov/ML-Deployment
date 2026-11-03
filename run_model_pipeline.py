from src.ml_deployment.pipeline import Config, Pipeline


def main():
    cfg = Config()
    print(cfg)

    p = Pipeline(cfg)
    p.load_data()          # train/val split, with the test-set guards
    p.build_transforms()   # image -> tensor recipe
    p.build_loaders()      # needs the data frames and the transform
    p.build_model()        # independent of the data stages, but kept last for clarity
    p.build_optimizer()    
    print("train:", len(p.train_df), "| val:", len(p.val_df), "| device:", p.device)

    #p.train()
    # p.evaluate()


if __name__ == "__main__":
    main()