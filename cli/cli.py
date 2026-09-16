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
    
    eval_parser = subparsers.add_parser("trainval", help="Train model on full train+val and evaluate on test set")
    eval_parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["LogisticRegression", "RandomForest"],
        help="Which model to train for production and evaluate"
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
            
            elif args.command == "eval":
                import os
                import glob
                
                baseline_dir = "artifacts/baseline"
                if not os.path.exists(baseline_dir):
                    print("no basemodels found.\n")
                    continue
                    
                model_files = glob.glob(os.path.join(baseline_dir, "*_best.joblib"))
                
                if not model_files:
                    print("no model files (*_best.joblib) found.\n")
                    continue
                
                available_models = []
                print("\nAvaliable baseline models:")
                print("-" * 30)
                for i, path in enumerate(model_files, 1):
                    filename = os.path.basename(path)
                    model_name = filename.replace("_best.joblib", "").capitalize()
                    if "logistic" in model_name.lower():
                        model_name = "LogisticRegression"
                    elif "random" in model_name.lower():
                        model_name = "RandomForest"
                        
                    available_models.append((i, model_name, path))
                    print(f"[{i}] {model_name} (Fil: {filename})")
                print("-" * 30)
                
                choice = input("Select the number of the ,odel you would like to train: ").strip()
                
                try:
                    selected_idx = int(choice) - 1
                    chosen_model_name = available_models[selected_idx][1]
                    
                    print(f"Model: {chosen_model_name}. is in training")
                    
                    from training.train_production import run_production_training
                    
                    default_params = {}
                    if chosen_model_name == "LogisticRegression":
                        default_params = {'C': 10.0}
                    elif chosen_model_name == "RandomForest":
                        default_params = {'n_estimators': 50, 'max_depth': 10}
                        
                    run_production_training(model_name=chosen_model_name, custom_params=default_params)
                    
                except (ValueError, IndexError):
                    print("Error.\n")
            
        except SystemExit:
            continue
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()