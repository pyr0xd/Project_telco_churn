import argparse
import sys

# python -m cli.cli

def create_parser():
    """Create all parsers."""
    parser = argparse.ArgumentParser(description="CLI tool for model training interface", prog="")
    subparsers = parser.add_subparsers(dest="command", help="choose command")

    train_parser = subparsers.add_parser("train", help="Run train/val with GridSearch and MLflow-experiment")
    train_parser.add_argument(
        "--models", 
        nargs="+", 
        default=["LogisticRegression", "RandomForest"], 
        help="Models that will be trained"
    )
    
    runs_parser = subparsers.add_parser("runs", help="Previous models and params MLflow")
    runs_parser.add_argument(
        "--limit", 
        type=int, 
        default=5, 
        help="Previous runs (standard: 5)"
    )

    subparsers.add_parser("fetch", help="Fetch dataset from database")

    return parser

def main():
    parser = create_parser()
    print("================================================================")
    print(" Welcome to the CLI model training interface!")
    print(" Commandlist:")
    print(" fetch - fetch the data from datase")
    print(" train - train a model")
    print(" runs - previous model training runs")
    print(" ")
    print(" help - get command list")
    print(" exit")
    print("================================================================")

    while True:
        try:
            line = input("churn-cli> ").strip()
            if not line:
                continue
            
            parts = line.split()
            
            if parts[0] == "exit":
                print("Exiting CLI")
                break
                
            if parts[0] == "help":
                parser.print_help()
                continue

            args = parser.parse_args(parts)

            if args.command == "train":
                print(f"Starting training for models: {args.models}...")
                from training.train_test import run_training
                run_training(selected_models=args.models)
                print("Training complete! Models saved and logged to MLflow.\n")
                
            elif args.command == "fetch":
                print("Fetching data from the database")
                from training.dataset_load import fetch_dataset
                df = fetch_dataset()
                print(f"Completted! Fetched {len(df)} rows.\n")
            
            elif args.command == "runs":
                print("fetching MLflow history...")
                try:
                    import mlflow
                    from mlflow.tracking import MlflowClient
                    
                    runs = mlflow.search_runs(experiment_names=["basemodel_test"])
                    
                    if runs.empty:
                        print("No previous runs.\n")
                    else:
                        print(f"\nLatest models (Max {args.limit} st):")
                        print("-" * 70)
                        
                        cols_to_show = ['run_id', 'params.model_type', 'metrics.val_f1_score', 'metrics.val_accuracy']
                        available_cols = [c for c in cols_to_show if c in runs.columns]
                        
                        print(runs[available_cols].head(args.limit).to_string(index=False))
                        print("-" * 70 + "\n")
                        
                except Exception as e:
                    print(f"Could not find MLflow history: {e}\n")
            
        except SystemExit:
            continue
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()